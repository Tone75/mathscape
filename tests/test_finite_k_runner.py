"""Deterministic runner contracts; no smoke/main or random samples are drawn."""
import json
import signal

import numpy as np
import pytest

from scripts import run_finite_k_benchmark as runner


def saved_fixture(R=1):
    samples=np.array([np.linspace(-.7,1.3,20)+i*.01 for i in range(R)])
    return dict(samples=samples,outer_id=np.arange(R),sample_sha256=np.array([runner.sample_hash(x).encode() for x in samples],dtype="S64"))


def fixed_bootstrap(sample,dist,**kw):
    # A deterministic endpoint fixture, not a resampling experiment.
    scale=(1+.03*kw['replicate_id']+.02*kw['inner_id'])
    means=sample.mean()+np.linspace(-.25,.25,kw['Bmax'])*scale
    if kw['return_indices']:
        indices=np.broadcast_to(np.arange(20,dtype=np.uint8),(kw['Bmax'],20)).copy()
        return means,indices
    return means


def tiny_cell(**kw):
    return runner.run_cell('gamma',0,saved_fixture(),R=1,J=2,K=8,B=99,
                           namespace=runner.SMOKE_NAMESPACE,bootstrap_fn=fixed_bootstrap,**kw)


def raw_original(panel):
    return {name:panel[name][0] for name in (*runner.DIAGNOSTICS,'valid','tail_warning')}


def test_tiny_injected_cell_schema_counts_B_and_cached_context(monkeypatch):
    original=runner.prepare_sample;calls=[]
    def prepare(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(runner,'prepare_sample',prepare)
    panel,tables=tiny_cell()
    assert calls==[1]
    assert panel['lower'].shape==(1,2,3,8) and panel['observed_streams'].all()
    assert len(tables['originals'])==3 and len(tables['original_K'])==6
    assert len(tables['replicates'])==6 and len(tables['groups'])==24 and len(tables['K_rows'])==12
    assert all(row['B']==99 for rows in tables.values() for row in rows)
    t=tables['originals'][0]
    assert t['total_inner']==t['valid_inner']==1 and t['invalid_inner']==0
    assert t['Q30']==t['Q120']==t['d']==t['W']==0
    assert all(row['K'] in (2,8) for row in tables['K_rows'])


@pytest.mark.parametrize('failure',['missing_t','nan','reversed','overflow'])
def test_derived_counts_follow_core_strict_required_slot_validity(failure):
    panel,_=tiny_cell();raw=raw_original(panel)
    if failure=='missing_t':
        raw['valid'][1,0,7]=False;raw['lower'][1,0,7]=np.nan
    elif failure=='nan':raw['lower'][1,2,7]=np.nan
    elif failure=='reversed':raw['lower'][1,2,7]=2;raw['upper'][1,2,7]=1
    else:raw['lower'][1,2,7]=-1e308;raw['upper'][1,2,7]=1e308
    se=np.std(panel['samples'][0],ddof=1)/np.sqrt(20)
    result=runner.original_metrics(raw,se,group_size=2,expected_replicates=2)
    tables=runner.derived_tables(result,raw,se,'gamma',0,0,runner.sample_hash(panel['samples'][0]),1,2,8,B=99)
    method='t' if failure=='missing_t' else 'bca'
    full=next(r for r in tables['original_K'] if r['method']==method and r['K']==8)
    rep=next(r for r in tables['K_rows'] if r['method']==method and r['K']==8 and r['replicate_id']==1)
    assert not full['complete'] and full['invalid_inner']==1 and rep['invalid_inner']==1
    assert full['valid_inner']==(0 if method=='t' else 15)
    assert np.isnan(full['A'])
    assert next(r for r in tables['originals'] if r['method']=='percentile')['complete']


def test_mid_first_original_stop_retains_completed_streams_and_uncomputed_mask(tmp_path):
    calls=0;progress={}
    def stop():
        nonlocal calls
        calls+=1
        if calls==5:raise runner.BudgetExceeded('fixture budget')
    with pytest.raises(runner.BudgetExceeded):tiny_cell(check=stop,progress=progress)
    assert progress['observed_original_count']==0 and progress['retained_original_count']==1
    assert progress['observed_stream_count']==3
    assert progress['panel']['observed_streams'].sum()==3
    assert progress['panel']['valid'].sum()==9
    assert not progress['tables']['originals']
    panel={**progress['panel'],'reason_labels':np.array(progress['reason_labels'],dtype='U96')}
    target=tmp_path/'partial.npz';runner.save_panel(target,panel,'gamma',0,2,8,99)
    with np.load(target,allow_pickle=False) as stored:
        assert stored['observed_streams'].sum()==3
        assert np.isnan(stored['lower'][0,0,:,3:]).all()
    with pytest.raises(FileExistsError):runner.save_panel(target,panel,'gamma',0,2,8,99)


def test_budget_thresholds_and_publication_reserve_are_strict(tmp_path,monkeypatch):
    limit=runner.POLICY['output_bytes']-runner.POLICY['public_overhead_reserve_bytes']-65536
    runner.enforce_budget(293.99,2*1024**3-1,limit-1)
    for wall,rss,size in ((294,0,0),(0,2*1024**3,0),(0,0,limit)):
        with pytest.raises(runner.BudgetExceeded):runner.enforce_budget(wall,rss,size)
    monkeypatch.setattr(runner,'directory_bytes',lambda _:100)
    runner.require_output_capacity(tmp_path,limit-101)
    with pytest.raises(runner.BudgetExceeded):runner.require_output_capacity(tmp_path,limit-100)
    payload={'a':np.ones((100,20)), 'label':'fixture'}
    assert runner.npz_size_bound(payload)>sum(np.asarray(v).nbytes for v in payload.values())


def test_panel_capacity_is_checked_before_exclusive_open(tmp_path,monkeypatch):
    panel,_=tiny_cell();target=tmp_path/'panel.npz'
    monkeypatch.setattr(runner,'require_output_capacity',lambda *a:(_ for _ in ()).throw(runner.BudgetExceeded('full')))
    with pytest.raises(runner.BudgetExceeded):runner.save_panel(target,panel,'gamma',0,2,8,99,capacity_output=tmp_path)
    assert not target.exists()


def complete_rows():
    rows=[]
    for d in runner.DISTRIBUTIONS:
        for seed in (0,1,2):
            for outer in range(10):
                rows.append(dict(distribution=d,master_seed=seed,outer_id=outer,method='bca',complete=True,
                    d=.0001*(outer-4.5),W=-.01-.0001*outer,ratio_Q120_Q30=.3+.01*outer,
                    Q30=.02+.001*outer,Q120=(.3+.01*outer)*(.02+.001*outer),
                    **{'mean_'+gap:.0001*(outer-4.5) for gap in runner.GAPS}))
    return rows


def test_confirmatory_original_N_local_missing_and_no_seed_gate():
    rows=complete_rows();result=runner.assemble_hypotheses(rows)
    assert result['confirmatory_intervals']==4 and not result['seed_direction_gate']
    assert result['hypotheses']['H1']['status']=='supported'
    assert result['hypotheses']['H2']['status']=='supported'
    assert result['quantities']['H1_gamma']['N']==30
    rows=[r for r in rows if not (r['distribution']=='gamma' and r['master_seed']==2 and r['outer_id']==9)]
    result=runner.assemble_hypotheses(rows)
    assert result['quantities']['H1_gamma']['N']==30 and result['quantities']['H1_gamma']['finite_count']==29
    assert result['hypotheses']['H1']['gamma']=='inconclusive'
    assert result['hypotheses']['H1']['lognormal']=='supported'


def test_ratio_references_are_distinct_and_undefined_algebra_is_not_failure():
    originals=complete_rows()
    reps=[]
    for r in originals:
        reps.append({**r,**{g:r['mean_'+g] for g in runner.GAPS},**{f:0. for f in runner.RESIDUALS},
                     'algebra_ok':True,'reason':''})
    reps[0]['algebra_ok']=False;reps[0]['reason']='missing_endpoint'
    reps[1]['algebra_ok']=False;reps[1]['reason']='algebraic_identity_failed'
    row=next(r for r in runner.decomposition_summary(originals,reps) if r['distribution']=='gamma' and r['master_seed']=='pooled' and r['method']=='bca')
    assert row['algebra_failed_replicates']==row['algebra_unchecked_replicates']==1
    assert row['ratio_planned_originals']==row['ratio_finite_originals']==30
    assert row['mean_individual_Q120_Q30']!=row['ratio_mean_Q120_over_mean_Q30']
    originals[0]['ratio_Q120_Q30']=np.nan
    row=next(r for r in runner.decomposition_summary(originals,reps) if r['distribution']=='gamma' and r['master_seed']=='pooled' and r['method']=='bca')
    assert np.isnan(row['mean_individual_Q120_Q30']) and row['ratio_finite_originals']==29
    assert np.isfinite(row['conditional_mean_individual_Q120_Q30'])


def test_cleanup_disarms_timer_before_ignoring_TERM(monkeypatch):
    events=[]
    monkeypatch.setattr(runner.signal,'setitimer',lambda *a:events.append(('timer',a)))
    monkeypatch.setattr(runner.signal,'signal',lambda *a:events.append(('signal',a)))
    runner._begin_cleanup()
    assert events==[('timer',(signal.ITIMER_REAL,0)),('signal',(signal.SIGTERM,signal.SIG_IGN))]


def test_budget_stop_finally_preserves_first_partial_panel_and_resources(tmp_path,monkeypatch):
    """Exercise orchestration only with injected fixed data and a three-slot stop."""
    source=runner.run_cell;saved=saved_fixture();calls=0
    def interrupted(dist,seed,unused,**kw):
        def stop():
            nonlocal calls
            calls+=1
            if calls==5:raise runner.BudgetExceeded('fixture budget')
        return source(dist,seed,saved,R=1,J=20,K=120,B=4999,check=stop,
                      namespace=runner.SMOKE_NAMESPACE,bootstrap_fn=fixed_bootstrap,progress=kw['progress'])
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'verify_committed_inputs',lambda *a:None)
    monkeypatch.setattr(runner,'load_inputs',lambda *a,**kw:{('gamma',0):saved})
    monkeypatch.setattr(runner,'run_cell',interrupted)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,question_id='RQ-M07',design=runner.DESIGN,policy=runner.POLICY,inputs={})))
    with pytest.raises(runner.BudgetExceeded):runner.run_benchmark(manifest,tmp_path/'evidence','a'*40)
    output=tmp_path/'evidence'
    resources=json.loads((output/'resources.json').read_text())
    assert resources['status']=='stopped' and resources['completed_cells']==0
    assert resources['completed_originals']==0 and resources['partial_retained_original_count']==1
    assert resources['observed_inner_streams']==3 and resources['observed_logical_intervals']==9
    with np.load(output/'panels/gamma_seed0_partial.npz',allow_pickle=False) as stored:
        assert stored['completed_original_count']==0 and stored['observed_stream_count']==3
        assert stored['observed_streams'].sum()==3
    with pytest.raises(FileExistsError):runner.run_benchmark(manifest,output,'a'*40)
