"""Injected deterministic runner/stop/schema fixtures; no scientific draws."""
import json
import signal

import numpy as np
import pytest

from scripts import run_allocation_benchmark as runner


def saved_fixture(O=1):
    samples=np.array([np.linspace(-.7,1.3,20)+i*.01 for i in range(O)])
    return dict(samples=samples,outer_id=np.arange(O),sample_sha256=np.array([runner.sample_hash(x).encode() for x in samples],dtype='S64'))


def fixed_bootstrap(sample,dist,**kw):
    # Fixed analytical endpoint vector, never a resampling experiment.
    scale=1+.03*kw['budget_repeat_id']+.02*kw['stream_id']
    means=sample.mean()+np.linspace(-.25,.25,kw['Bmax'])*scale
    if kw['return_indices']:return means,np.broadcast_to(np.arange(20,dtype=np.uint8),(kw['Bmax'],20)).copy()
    return means


def tiny_cell(**kw):
    return runner.run_cell('gamma',0,saved_fixture(),O=1,R=2,streams=12,B=99,namespace=runner.SMOKE_NAMESPACE,bootstrap_fn=fixed_bootstrap,**kw)


def raw_original(panel):return {key:panel[key][0] for key in (*runner.DIAGNOSTICS,'valid','tail_warning')}


def test_exact_shared_budget_schema_rows_counts_B_and_cached_original_context(monkeypatch):
    original=runner.prepare_sample;calls=[];draws=[]
    def prepare(*a,**kw):calls.append(1);return original(*a,**kw)
    def fixed(*a,**kw):draws.append((kw['budget_repeat_id'],kw['stream_id']));return fixed_bootstrap(*a,**kw)
    monkeypatch.setattr(runner,'prepare_sample',prepare)
    panel,tables=runner.run_cell('gamma',0,saved_fixture(),O=1,R=2,streams=12,B=99,namespace=runner.SMOKE_NAMESPACE,bootstrap_fn=fixed)
    assert calls==[1] and draws==[(r,k) for r in range(2) for k in range(12)]
    assert panel['lower'].shape==(1,2,3,12) and panel['observed_streams'].all()
    assert {key:len(value) for key,value in tables.items()}==dict(original_allocations=9,original_comparisons=3,budgets=18,groups=42,decompositions=6)
    assert all(row['B']==99 and row['R']==2 and row['originals_per_cell']==1 for rows in tables.values() for row in rows)
    assert {(r['allocation'],r['J'],r['K']) for r in tables['budgets']}=={('P30',4,3),('P60',2,6),('P120',1,12)}
    for allocation,expected in (('P30',[(0,3),(3,6),(6,9),(9,12)]),('P60',[(0,6),(6,12)]),('P120',[(0,12)])):
        rows=[r for r in tables['groups'] if r['method']=='bca' and r['budget_repeat_id']==0 and r['allocation']==allocation]
        assert [(r['group_start'],r['group_stop']) for r in rows]==expected
        budget=next(r for r in tables['budgets'] if r['method']=='bca' and r['budget_repeat_id']==0 and r['allocation']==allocation)
        assert budget['T']==pytest.approx(np.mean([r['A'] for r in rows]))
        assert budget['Vbar']==pytest.approx(np.mean([r['V'] for r in rows]))
    t=next(r for r in tables['original_comparisons'] if r['method']=='t')
    assert t['total_streams']==t['valid_streams']==1 and t['invalid_streams']==0
    assert t['d']==t['L']==t['U']==t['Q30']==t['Q120']==0 and np.isnan(t['ratio_Q120_Q30'])


@pytest.mark.parametrize('failure',['missing_t','nan','reversed','overflow'])
def test_strict_counts_follow_all_required_slots_not_first_t_slot(failure):
    panel,_=tiny_cell();raw=raw_original(panel)
    if failure=='missing_t':raw['valid'][1,0,11]=False;raw['lower'][1,0,11]=np.nan
    elif failure=='nan':raw['lower'][1,2,11]=np.nan
    elif failure=='reversed':raw['lower'][1,2,11]=2;raw['upper'][1,2,11]=1
    else:raw['lower'][1,2,11]=-1e308;raw['upper'][1,2,11]=1e308
    se=np.std(panel['samples'][0],ddof=1)/np.sqrt(20)
    result=runner.original_metrics(raw,se,group_size=3,expected_repeats=2)
    tables=runner.derived_tables(result,raw,se,'gamma',0,0,runner.sample_hash(panel['samples'][0]),1,2,12,B=99)
    method='t' if failure=='missing_t' else 'bca'
    rows=[r for r in tables['original_allocations'] if r['method']==method]
    assert all(not r['complete'] and r['invalid_streams']==1 and np.isnan(r['M']) for r in rows)
    assert all(r['valid_streams']==(0 if method=='t' else 23) for r in rows)
    assert next(r for r in tables['original_comparisons'] if r['method']=='percentile')['complete']


def comparisons_fixture():
    rows=[]
    for dist in runner.DISTRIBUTIONS:
        for seed in (0,1,2):
            for outer in range(10):
                q=.02+.001*outer;qfull=q*(1+.001*(outer-4.5))
                rows.append(dict(distribution=dist,master_seed=seed,outer_id=outer,method='bca',complete=True,d=.0001*(outer-4.5),L=qfull-.8*q,U=qfull-1.25*q,
                    Q30=q,Q120=qfull,ratio_Q120_Q30=qfull/q,mean_Jgap=.001+.0001*outer,mean_Pgap=-.001,mean_delta=.0001*outer))
    return rows


def test_six_original_level_primary_intervals_and_multiplicative_band_no_seed_gate():
    rows=comparisons_fixture();h=runner.assemble_hypotheses(rows)
    assert h['confirmatory_intervals']==len(h['quantities'])==6 and h['decision_confidence']==.995
    assert not h['seed_direction_gate'] and h['P60']=='descriptive only'
    assert h['H2_variance_ratio_band']==[.8,1.25]
    assert h['hypotheses']['H1']['status']==h['hypotheses']['H2']['status']=='supported'
    assert all(x['N']==30 and x['df']==29 for x in h['quantities'].values())
    for seed in h['seed_details']:assert all(q['N']==10 for k,q in seed.items() if k!='master_seed')
    rows=[r for r in rows if not (r['distribution']=='gamma' and r['master_seed']==2 and r['outer_id']==9)]
    h=runner.assemble_hypotheses(rows)
    assert h['quantities']['H1_gamma']['N']==30 and h['quantities']['H1_gamma']['finite_count']==29
    assert h['hypotheses']['H1']['gamma']==h['hypotheses']['H2']['gamma']=='inconclusive'
    assert h['hypotheses']['H1']['lognormal']==h['hypotheses']['H2']['lognormal']=='supported'


def test_unrelated_method_invalid_does_not_poison_BCa_and_denominator_zero_is_hold():
    rows=comparisons_fixture();rows.append(dict(rows[0],method='percentile',complete=False,d=np.nan,L=np.nan,U=np.nan))
    assert runner.assemble_hypotheses(rows)['hypotheses']['H2']['status']=='supported'
    for r in rows:
        if r['distribution']=='gamma' and r['method']=='bca':r['Q30']=0
    h=runner.assemble_hypotheses(rows)
    assert h['population_decisions']['gamma']['H2']['status']=='inconclusive'


def test_mid_original_partial_preserves_observed_mask_and_never_overwrites(tmp_path):
    progress={};calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls==5:raise runner.BudgetExceeded('fixturestop')
    with pytest.raises(runner.BudgetExceeded):tiny_cell(check=stop,progress=progress)
    assert progress['completed_original_count']==0 and progress['retained_original_count']==1 and progress['observed_stream_count']==3
    panel={**progress['panel'],'reason_labels':np.array(progress['reason_labels'],dtype='U96')};p=tmp_path/'partial.npz'
    runner.save_panel(p,panel,'gamma',0,2,12,99)
    with np.load(p,allow_pickle=False) as z:
        assert z['observed_streams'].sum()==3 and z['completed_original_count']==0
        assert z['O']==10 and z['R']==2 and z['streams_per_budget']==12 and z['B']==99
        assert np.isnan(z['lower'][0,0,:,3:]).all()
    with pytest.raises(FileExistsError):runner.save_panel(p,panel,'gamma',0,2,12,99)


def test_prewrite_capacity_wall_rss_and_public_reserve(tmp_path,monkeypatch):
    limit=runner.POLICY['output_bytes']-runner.POLICY['public_overhead_reserve_bytes']-65536
    runner.enforce_budget(293.99,2*1024**3-1,limit-1)
    for wall,rss,size in ((294,0,0),(0,2*1024**3,0),(0,0,limit)):
        with pytest.raises(runner.BudgetExceeded):runner.enforce_budget(wall,rss,size)
    monkeypatch.setattr(runner,'directory_bytes',lambda _:100)
    runner.require_output_capacity(tmp_path,limit-101)
    with pytest.raises(runner.BudgetExceeded):runner.require_output_capacity(tmp_path,limit-100)
    panel,_=tiny_cell();target=tmp_path/'blocked.npz'
    monkeypatch.setattr(runner,'require_output_capacity',lambda *a:(_ for _ in ()).throw(runner.BudgetExceeded('full')))
    with pytest.raises(runner.BudgetExceeded):runner.save_panel(target,panel,'gamma',0,2,12,99,capacity_output=tmp_path)
    assert not target.exists()


def test_stopped_orchestration_finally_retains_first_partial_panel_and_resources(tmp_path,monkeypatch):
    source=runner.run_cell;saved=saved_fixture();calls=0
    def interrupted(dist,seed,unused,**kw):
        def stop():
            nonlocal calls
            calls+=1
            if calls==5:raise runner.BudgetExceeded('fixturestop')
        return source(dist,seed,saved,O=1,R=20,streams=120,B=4999,check=stop,namespace=runner.SMOKE_NAMESPACE,bootstrap_fn=fixed_bootstrap,progress=kw['progress'])
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'verify_committed_inputs',lambda *a:None)
    monkeypatch.setattr(runner,'load_inputs',lambda *a,**kw:{('gamma',0):saved});monkeypatch.setattr(runner,'run_cell',interrupted)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(schema_version=1,question_id='RQ-M08',design=runner.DESIGN,policy=runner.POLICY,inputs={})))
    with pytest.raises(runner.BudgetExceeded):runner.run_benchmark(path,tmp_path/'evidence','a'*40)
    record=json.loads((tmp_path/'evidence/resources.json').read_text())
    assert record['status']=='stopped' and record['completed_cells']==record['completed_originals']==0
    assert record['partial_retained_original_count']==1 and record['observed_inner_streams']==3 and record['observed_logical_intervals']==9
    assert record['endpoint_generation_inference_seconds']>=0 and record['allocation_aggregation_flatten_seconds']==0
    with pytest.raises(FileExistsError):runner.run_benchmark(path,tmp_path/'evidence','a'*40)


def test_undefined_algebra_not_counted_as_failure_and_Q_ratios_distinct():
    comparisons=comparisons_fixture();budgets=[]
    for r in comparisons:budgets.append({**r,'Jgap':r['mean_Jgap'],'Pgap':r['mean_Pgap'],'delta':r['mean_delta'],'residual':0.,'algebra_ok':True,'reason':''})
    budgets[0].update(algebra_ok=False,reason='endpoint_invalid_or_nonfinite');budgets[1].update(algebra_ok=False,reason='algebraic_identity_failed')
    row=next(r for r in runner.decomposition_summary(comparisons,budgets) if r['distribution']=='gamma' and r['master_seed']=='pooled' and r['method']=='bca')
    assert row['algebra_failed_repeats']==row['algebra_unchecked_repeats']==1
    assert row['ratio_planned_originals']==row['ratio_finite_originals']==30
    assert row['mean_individual_Q120_Q30']!=row['ratio_mean_Q120_over_mean_Q30']
    comparisons[0]['ratio_Q120_Q30']=np.nan
    row=next(r for r in runner.decomposition_summary(comparisons,budgets) if r['distribution']=='gamma' and r['master_seed']=='pooled' and r['method']=='bca')
    assert np.isnan(row['mean_individual_Q120_Q30']) and row['ratio_finite_originals']==29
    assert np.isfinite(row['conditional_mean_individual_Q120_Q30'])


def test_cleanup_disarms_timer_before_ignoring_TERM(monkeypatch):
    events=[];monkeypatch.setattr(runner.signal,'setitimer',lambda *a:events.append(('timer',a)));monkeypatch.setattr(runner.signal,'signal',lambda *a:events.append(('signal',a)))
    runner._begin_cleanup()
    assert events==[('timer',(signal.ITIMER_REAL,0)),('signal',(signal.SIGTERM,signal.SIG_IGN))]
