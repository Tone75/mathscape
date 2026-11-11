"""Injected deterministic runner/stop/schema fixtures; no scientific draws."""
import json
import signal

import numpy as np
import pytest

from scripts import run_small_sample_benchmark as runner
from mathscape import small_sample as engine


@pytest.fixture(scope='module')
def old_reference():
    return engine.load_reference(root=runner.ROOT)


def fake_original(dist,n=5,master_seed=0,outer_id=0,namespace=runner.MAIN_NAMESPACE):
    """Fake API inputs, never samples from a PRNG or a pilot cohort."""
    class Fixed:
        def gamma(self,shape,scale,size):return np.linspace(.15,2.3,size)+.1*master_seed+.003*outer_id
        def standard_normal(self,size):return np.linspace(-1.7,1.9,size)+.04*master_seed+.002*outer_id
    return engine.draw_original(dist,n,master_seed,outer_id,namespace,rng=Fixed())


@pytest.fixture(scope='module')
def fake_cohort(old_reference):
    return engine.generate_cohort(old_reference,draw_fn=fake_original)


def contract_manifest():
    return dict(generation={'rule':'deterministic fixture only'},generation_registry=engine.generation_design(),
                source_files_sha256={name:runner.sha256(runner.ROOT/name) for name in ('mathscape/small_sample.py','mathscape/replication.py','mathscape/matched_moments.py')})


def saved_fixture(O=1):
    samples=np.array([np.linspace(-.7,1.3,5)+i*.01 for i in range(O)])
    return dict(samples=samples,outer_id=np.arange(O),sample_sha256=np.array([runner.sample_hash(x).encode() for x in samples],dtype='S64'))


def fixed_bootstrap(sample,dist,**kw):
    # Fixed analytical endpoint vector, never a resampling experiment.
    scale=1+.03*kw['budget_repeat_id']+.02*kw['stream_id']
    means=sample.mean()+np.linspace(-.25,.25,kw['Bmax'])*scale
    if kw['return_indices']:return means,np.broadcast_to(np.arange(5,dtype=np.uint8),(kw['Bmax'],5)).copy()
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
    assert {key:len(value) for key,value in tables.items()}==dict(original_allocations=9,original_comparisons=3,budgets=18,groups=42,decompositions=6,diagnostics=3,reasons=3)
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
    se=np.std(panel['samples'][0],ddof=1)/np.sqrt(5)
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
    monkeypatch.setattr(runner,'load_reference',lambda *a,**kw:{'cells':{('gamma',0):saved}});monkeypatch.setattr(runner,'run_cell',interrupted)
    def generated(reference,**kw):kw['progress'].update(generated_originals=60);return {('gamma',0):saved}
    def persist(directory,*a,**kw):directory.mkdir();(directory/'receipt.json').write_text('{}')
    monkeypatch.setattr(runner,'generate_cohort',generated);monkeypatch.setattr(runner,'persist_cohort',persist)
    monkeypatch.setattr(runner,'readback_cohort',lambda *a:({('gamma',0):saved},{}))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(schema_version=1,question_id='RQ-M10',design=runner.DESIGN,policy=runner.POLICY,reference={})))
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


def test_full_fake_original_cohort_persistence_receipt_readback_before_any_inner_rng(tmp_path,old_reference,fake_cohort,monkeypatch):
    monkeypatch.setattr(engine,'rng_for_original',lambda *a,**k:pytest.fail('No original PRNG in fixture/readback'))
    monkeypatch.setattr(engine,'rng_for_panel',lambda *a,**k:pytest.fail('No inner PRNG in fixture/readback'))
    m=contract_manifest();directory=tmp_path/'originals'
    receipt=runner.persist_cohort(directory,fake_cohort,old_reference,'a'*40,m)
    restored,read=runner.readback_cohort(directory,old_reference,m)
    assert read==receipt and receipt['original_samples']==60 and len(receipt['records'])==60
    assert receipt['generation_contract_sha256']==runner.generation_contract(m)
    assert set(receipt['original_files_sha256'])=={f'{d}_seed{s}.npz' for d,s in runner.fixed_cells()}
    for key in fake_cohort:
        for field in fake_cohort[key]:np.testing.assert_array_equal(restored[key][field],fake_cohort[key][field])
        assert not restored[key]['samples'].flags.writeable
        assert restored[key]['original_keys'].shape==(10,6) and restored[key]['raw'].shape==(10,5)
    with pytest.raises(FileExistsError):runner.persist_cohort(directory,fake_cohort,old_reference,'a'*40,m)


@pytest.mark.parametrize('tamper',['sample_file','receipt_file_set','generation_rule','generation_source','receipt_record'])
def test_reuse_originals_rejects_hash_rules_sources_and_receipt_tampering(tmp_path,old_reference,fake_cohort,tamper):
    m=contract_manifest();directory=tmp_path/'originals';runner.persist_cohort(directory,fake_cohort,old_reference,'a'*40,m)
    if tamper=='sample_file':
        path=directory/'gamma_seed0.npz';path.write_bytes(path.read_bytes()+b'tampered')
    elif tamper=='generation_rule':m['generation']['rule']='changedpopulation'
    elif tamper=='generation_source':m['source_files_sha256']['mathscape/small_sample.py']='0'*64
    else:
        path=directory/'receipt.json';r=json.loads(path.read_text())
        if tamper=='receipt_file_set':r['original_files_sha256']['unexpected.npz']=r['original_files_sha256'].pop('gamma_seed0.npz')
        else:r['records'][0]['spawn_key'][-1]=1
        path.write_text(json.dumps(r))
    with pytest.raises(ValueError):runner.readback_cohort(directory,old_reference,m)


def test_prereg_source_failure_prevents_even_original_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'verify_committed_inputs',lambda *a:(_ for _ in ()).throw(ValueError('unreviewedsource')))
    monkeypatch.setattr(runner,'generate_cohort',lambda *a,**kw:pytest.fail('must not generate before source verification'))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(schema_version=1,question_id='RQ-M10',design=runner.DESIGN,policy=runner.POLICY,reference={})))
    with pytest.raises(ValueError,match='unreviewedsource'):runner.run_benchmark(path,tmp_path/'evidence','a'*40)
    assert not (tmp_path/'evidence').exists()


def test_full_original_receipt_readback_is_a_hard_inner_generation_barrier(tmp_path,monkeypatch,old_reference,fake_cohort):
    m=contract_manifest();m.update(schema_version=1,question_id='RQ-M10',design=runner.DESIGN,policy=runner.POLICY,reference=old_reference['reference'])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m));events=[]
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'verify_committed_inputs',lambda *a:events.append('verified'))
    monkeypatch.setattr(runner,'load_reference',lambda *a,**kw:old_reference)
    def generate(*a,**kw):events.append('all60fake');kw['progress'].update(generated_originals=60,cells=fake_cohort);return fake_cohort
    monkeypatch.setattr(runner,'generate_cohort',generate)
    persist=runner.persist_cohort
    def save(*a,**kw):result=persist(*a,**kw);events.append('persisted');return result
    monkeypatch.setattr(runner,'persist_cohort',save)
    def fail_read(*a,**kw):events.append('readbackfailed');raise ValueError('fixture_readback_failure')
    monkeypatch.setattr(runner,'readback_cohort',fail_read)
    monkeypatch.setattr(runner,'run_cell',lambda *a,**kw:pytest.fail('No inner series before readback success'))
    with pytest.raises(ValueError,match='fixture_readback_failure'):runner.run_benchmark(path,tmp_path/'evidence','a'*40)
    assert events==['verified','all60fake','persisted','readbackfailed']
    resources=json.loads((tmp_path/'evidence/resources.json').read_text())
    assert resources['mother_samples_generated']==60 and not resources['original_receipt_readback_verified']
    assert resources['observed_inner_streams']==0 and (tmp_path/'evidence/originals/receipt.json').exists()


def test_partial_originals_are_preserved_without_any_redraw_or_inner_call(tmp_path,monkeypatch,old_reference,fake_cohort):
    m=contract_manifest();m.update(schema_version=1,question_id='RQ-M10',design=runner.DESIGN,policy=runner.POLICY,reference=old_reference['reference'])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'verify_committed_inputs',lambda *a:None);monkeypatch.setattr(runner,'load_reference',lambda *a,**kw:old_reference)
    def stopped(*a,**kw):
        partial={k:(v[:3] if v.ndim and k in ('samples','raw','sample_sha256','raw_sha256','outer_id','original_keys') else v) for k,v in fake_cohort[('gamma',0)].items()}
        kw['progress'].update(cells={('gamma',0):partial},generated_originals=3)
        raise runner.BudgetExceeded('fixed original stop')
    monkeypatch.setattr(runner,'generate_cohort',stopped);monkeypatch.setattr(runner,'run_cell',lambda *a,**kw:pytest.fail('must not run inner series'))
    with pytest.raises(runner.BudgetExceeded):runner.run_benchmark(path,tmp_path/'evidence','a'*40)
    with np.load(tmp_path/'evidence/originals_partial/gamma_seed0.npz',allow_pickle=False) as z:assert z['raw'].shape==(3,5)
    resources=json.loads((tmp_path/'evidence/resources.json').read_text())
    assert resources['mother_samples_generated']==resources['partial_originals_preserved']==3 and resources['observed_inner_streams']==0


def test_manifest_registry_contains_rules_keys_and_no_future_sample_hashes():
    registry=engine.generation_design()
    assert len(registry['originals'])==60 and registry['original_key_length']==6 and registry['inner_key_length']==7
    assert not registry['sample_hashes_known_before_generation']
    assert all(len(row['spawn_key'])==6 and row['spawn_key'][4:]==[0,0] for row in registry['originals'])


def test_existing_canonical_attempt_prevents_fresh_output_hidden_redraw(tmp_path,monkeypatch,old_reference):
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'verify_committed_inputs',lambda *a:None)
    monkeypatch.setattr(runner,'load_reference',lambda *a,**kw:old_reference)
    monkeypatch.setattr(runner,'generate_cohort',lambda *a,**kw:pytest.fail('No redraw after an existing attempt'))
    canonical=tmp_path/'canonical';canonical.mkdir();(canonical/'attempt.json').write_text('{}')
    path=canonical/'manifest.json';path.write_text(json.dumps(dict(schema_version=1,question_id='RQ-M10',design=runner.DESIGN,policy=runner.POLICY,reference={})))
    with pytest.raises(ValueError,match='explicit reuse'):runner.run_benchmark(path,tmp_path/'correction','a'*40)
    assert not (tmp_path/'correction').exists()


def retained_three(fake_cohort):
    cell=fake_cohort[('gamma',0)]
    return {('gamma',0):{key:(value[:3] if key in ('samples','raw','sample_sha256','raw_sha256','outer_id','original_keys') else value) for key,value in cell.items()}}


def test_partial_receipt_readback_resumes_only_missing_fixed_ids_without_redraw(tmp_path,old_reference,fake_cohort,monkeypatch):
    preserved=retained_three(fake_cohort);m=contract_manifest();directory=tmp_path/'partial'
    runner.persist_cohort(directory,preserved,old_reference,'a'*40,m,partial=True)
    restored,receipt=runner.readback_cohort(directory,old_reference,m,allow_partial=True)
    assert receipt['status']=='partial' and receipt['original_samples']==3
    calls=[];progress={}
    def missing_only(dist,**kw):
        identity=(dist,kw['master_seed'],kw['outer_id']);calls.append(identity)
        assert identity not in [('gamma',0,i) for i in range(3)]
        return fake_original(dist,**kw)
    monkeypatch.setattr(engine,'rng_for_original',lambda *a,**kw:pytest.fail('No actual RNG for correction fixture'))
    resumed=engine.generate_cohort(old_reference,draw_fn=missing_only,existing_cells=restored,progress=progress)
    assert len(calls)==57 and progress['reused_originals']==3 and progress['generated_originals']==57
    assert resumed[('gamma',0)]['samples'][:3].tobytes()==preserved[('gamma',0)]['samples'].tobytes()
    with pytest.raises(ValueError):runner.readback_cohort(directory,old_reference,m)


@pytest.mark.parametrize('collision',['duplicate','smoke_projection'])
def test_offending_original_raw_survives_invalid_partial_receipt_and_cannot_resume(tmp_path,old_reference,fake_cohort,collision):
    cell={key:value.copy() for key,value in retained_three(fake_cohort)[('gamma',0)].items()}
    if collision=='duplicate':raw=cell['raw'][0].copy();sample=cell['samples'][0].copy()
    else:
        sample=engine.smoke_projection(old_reference)[('gamma',0)]['samples'][0].copy();raw=sample+1
    cell['samples'][1]=sample;cell['raw'][1]=raw
    cell['sample_sha256'][1]=runner.sample_hash(sample).encode();cell['raw_sha256'][1]=runner.sample_hash(raw).encode()
    directory=tmp_path/'invalid';m=contract_manifest()
    receipt=runner.persist_cohort(directory,{('gamma',0):cell},old_reference,'a'*40,m,partial=True)
    assert receipt['status']=='invalid_preserved' and receipt['original_samples']==3
    with np.load(directory/'gamma_seed0.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['samples'][1],sample);np.testing.assert_array_equal(z['raw'][1],raw)
        assert z['outer_id'].tolist()==[0,1,2]
    with pytest.raises(ValueError,match='cannot be replaced or resumed'):runner.readback_cohort(directory,old_reference,m,allow_partial=True)


def test_half_tie_counts_are_uint16_actual_context_comparisons_and_non_gating(monkeypatch):
    context=runner.prepare_sample(saved_fixture()['samples'][0]);means=np.array([context['mean']-.1,context['mean'],context['mean'],context['mean']+.1])
    records=runner.infer_single(context,means)
    assert all((r['less_count'],r['equal_count'])==(1,2) for r in records.values())
    panel,tables=tiny_cell()
    assert panel['less_count'].dtype==panel['equal_count'].dtype==np.dtype('uint16')
    assert (panel['less_count']<=99).all() and (panel['equal_count']<=99).all()
    bca=next(r for r in tables['diagnostics'] if r['method']=='bca')
    assert bca['observed_streams']==24 and bca['invalid_streams']==bca['missing_streams']==0 and bca['tie_known_streams']==24
    infer=runner.infer_single
    def warning(*a,**kw):
        out=infer(*a,**kw);out['bca'].update(q_low=0.,q_high=1.,tau_low=0.,tau_high=0.,tail_warning=True);return out
    monkeypatch.setattr(runner,'infer_single',warning)
    _,tables=tiny_cell();bca=next(r for r in tables['diagnostics'] if r['method']=='bca')
    assert bca['warning_streams']==bca['q_low_zero_streams']==bca['q_high_one_streams']==24
    assert bca['invalid_streams']==0 and next(r for r in tables['original_comparisons'] if r['method']=='bca')['complete']


def test_partial_diagnostics_keep_missing_unknown_and_reasons_separate():
    panel,_=tiny_cell();panel['observed_streams'][0,1,11]=False
    panel['less_count'][0,1,:,11]=panel['equal_count'][0,1,:,11]=runner.UNKNOWN_COUNT
    panel['valid'][0,0,2,0]=False;panel['reason_codes'][0,0,2,0]=1
    rows,reasons=runner.diagnostic_rows(panel,0,['','z0_nonfinite'],'gamma',0,2,12,99)
    bca=next(r for r in rows if r['method']=='bca')
    assert bca['observed_streams']==23 and bca['missing_streams']==bca['invalid_streams']==1
    assert bca['tie_known_streams']==23
    assert any(r['status']=='invalid' and r['reason']=='z0_nonfinite' and r['streams']==1 for r in reasons)
    assert any(r['status']=='missing' and r['reason']=='not_observed' and r['streams']==1 for r in reasons)
    summary=next(r for r in runner.summarize_diagnostics(rows) if r['distribution']=='gamma' and r['master_seed']==0 and r['method']=='bca')
    assert summary['observed_originals']==1 and summary['absent_originals']==9 and summary['missing_originals']==10
    assert summary['invalid_originals']==1 and summary['complete_originals']==0 and summary['incomplete_originals']==10


def test_new_HTML_figure_declares_n5_and_distinct_sample_comparison_df(tmp_path,monkeypatch):
    import plotly.graph_objects as go
    captured=[];monkeypatch.setattr(go.Figure,'write_html',lambda self,*a,**kw:captured.append(self.to_plotly_json()))
    rows=[]
    for dist in runner.DISTRIBUTIONS:
        for allocation in runner.ALLOCATIONS:rows.append(dict(distribution=dist,method='bca',allocation=allocation,M=.04,Q=.001,M_decision_low=.03,M_decision_high=.05,Q_decision_low=.0005,Q_decision_high=.0015))
    runner.write_figure(rows,tmp_path/'figure')
    title=captured[0]['layout']['title']['text']
    assert all(text in title for text in ('RQ-M10','n5','alpha=.05','sample t df4','comparison df29'))
    assert len(captured[0]['data'])==4
    assert captured[0]['data'][0]['line']['color']==captured[0]['data'][1]['line']['color']
