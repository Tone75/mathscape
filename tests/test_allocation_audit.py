"""Fixed-array RQ-M08 audit contracts; never a random experiment."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy import stats
from scripts import audit_allocation as audit


def fixed_panel(o=2,r=20,k=120):
    samples=np.array([np.linspace(-.5,2,20)+oi*.01 for oi in range(o)])
    shape=(o,r,3,k);panel={key:np.full(shape,np.nan) for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high')}
    panel.update(samples=samples,sample_sha256=np.array([audit.sample_hash(sample).encode() for sample in samples],dtype='S64'),outer_id=np.arange(o,dtype=np.uint16),
                 budget_repeat_id=np.arange(r,dtype=np.uint8),stream_id=np.arange(k,dtype=np.uint8),methods=np.array(audit.METHODS),valid=np.ones(shape,bool),tail_warning=np.zeros(shape,bool),
                 reason_codes=np.zeros(shape,np.uint8),reason_labels=np.array(['','missing_endpoint'],dtype='U96'),observed_streams=np.ones((o,r,k),bool),
                 distribution=np.array('gamma'),master_seed=np.array(0),n=np.array(20),O=np.array(o),R=np.array(r),streams_per_budget=np.array(k),B=np.array(4999),alpha=np.array(.05),
                 namespace=np.array(audit.MAIN_NAMESPACE),input_accepted_head=np.array(audit.ACCEPTED_INPUT))
    for oi,sample in enumerate(samples):
        half=stats.t.ppf(.975,19)*np.std(sample,ddof=1)/np.sqrt(20);panel['lower'][oi,:,0]=sample.mean()-half;panel['upper'][oi,:,0]=sample.mean()+half
        for rep in range(r):
            for mi in (1,2):
                for group in range(4):
                    sl=slice(group*k//4,(group+1)*k//4);pattern=np.linspace(-1,1,k//4)*.01*(group+1)*(1+rep*.01+oi*.05)
                    panel['lower'][oi,rep,mi,sl]=-.4+pattern;panel['upper'][oi,rep,mi,sl]=.5+pattern*1.2
                panel['a'][oi,rep,mi]=0;panel['q_low'][oi,rep,mi]=.025;panel['q_high'][oi,rep,mi]=.975;panel['tau_low'][oi,rep,mi]=panel['tau_high'][oi,rep,mi]=4999*.025
                if mi==2:panel['z0'][oi,rep,mi]=0
    return panel


def test_audit_has_no_primary_aggregator_decision_or_previous_auditor_imports():
    tree=ast.parse(Path(audit.__file__).read_text());names=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any(name and ('mathscape' in name or 'run_allocation' in name or 'audit_finite' in name) for name in names)
    assert (audit.NUMERIC_RTOL,audit.NUMERIC_ATOL,audit.IDENTITY_RTOL,audit.IDENTITY_ATOL)==(2e-10,2e-12,2e-12,2e-12)


def test_square_root_then_average_group_sharing_and_two_SS_partitions():
    lo=np.tile([-1,1,-2,2,-3,3,-4,4.],(3,1));hi=lo+2;lo[0]=-.5;hi[0]=.5
    row=audit.budget_metrics(lo,hi,np.ones_like(lo,bool),np.zeros_like(lo,bool),1.)[2]
    allocations=row['allocations'];assert [item['J'] for item in allocations]==[4,2,1]
    assert [[item['group_start'] for item in a['groups']] for a in allocations]==[[0,2,4,6],[0,4],[0]]
    np.testing.assert_allclose([item['V'] for item in allocations[0]['groups']],[2,8,18,32])
    assert allocations[0]['T']==pytest.approx(2.5*np.sqrt(2)) and allocations[0]['T']!=pytest.approx(np.sqrt(15))
    assert allocations[0]['Vse']==allocations[0]['Vbar']==pytest.approx(15) and allocations[0]['Vse']!=pytest.approx(allocations[0]['Ase']**2)
    assert allocations[2]['T']==pytest.approx(np.sqrt(60/7))
    diag=row['diagnostics'];assert diag['Jgap']>0 and diag['Pgap']<0
    assert diag['delta']==pytest.approx(diag['Jgap']+diag['Pgap']) and not any('Rgap' in key for key in diag)
    for j in (4,2):
        for side in ('lower','upper'):
            assert abs(diag[f'ss{30 if j==4 else 60}_{side}_residual'])<=audit.IDENTITY_ATOL+audit.IDENTITY_RTOL*diag[f'ss{30 if j==4 else 60}_{side}_scale']
    t=audit.budget_metrics(lo,hi,np.ones_like(lo,bool),np.zeros_like(lo,bool),1.)[0]
    assert all(item['T']==item['Vbar']==0 for item in t['allocations'])


def test_R20_T_variance_and_paired_difference_not_independent_group_difference():
    row=next(row for row in audit.original_metrics(fixed_panel(o=1)) if row['method']=='bca')
    t30=np.array([rep['allocations'][0]['T'] for rep in row['repeats']]);t120=np.array([rep['allocations'][2]['T'] for rep in row['repeats']])
    assert row['Q30']==pytest.approx(np.var(t30,ddof=1));assert row['Q120']==pytest.approx(np.var(t120,ddof=1))
    assert row['d']==pytest.approx(np.mean(t120-t30));assert row['L']==pytest.approx(row['Q120']-.8*row['Q30']);assert row['U']==pytest.approx(row['Q120']-1.25*row['Q30'])
    assert row['total_inner']==20*120


@pytest.mark.parametrize('lo,hi,status',[(-.0024,.0024,'supported'),(-.0025,.0024,'inconclusive'),(-.0024,.0025,'inconclusive'),(.0026,.003,'not_supported'),(-.003,-.0026,'not_supported')])
def test_H1_strict_band_and_boundary_contact(lo,hi,status):
    assert audit.mean_status(dict(low=lo,high=hi,defined=True))==status


@pytest.mark.parametrize('llo,lhi,ulo,uhi,q,valid,status',[
    (.1,.2,-.2,-.1,.1,True,'supported'),(-.1,0,-.2,-.1,.1,True,'not_supported'),(.1,.2,0,.1,.1,True,'not_supported'),
    (0,.2,-.2,-.1,.1,True,'inconclusive'),(.1,.2,-.2,0,.1,True,'inconclusive'),(.1,.2,-.2,-.1,0,True,'inconclusive'),
    (-.1,0,-.2,-.1,.1,False,'inconclusive'),(.1,.2,-.2,-.1,np.nan,True,'inconclusive')])
def test_H2_two_valid_intervals_positive_averageQ_and_exact_gates(llo,lhi,ulo,uhi,q,valid,status):
    assert audit.variance_status(dict(low=llo,high=lhi,defined=valid),dict(low=ulo,high=uhi,defined=True),q)==status


def test_SE0_undefined_and_crosspopulation_clear_refutation_preserved():
    zero=audit.mean_interval([0.]*30);assert not zero['defined'] and zero['N']==30 and zero['mean']==zero['sd']==0
    assert audit.mean_status(zero)=='inconclusive' and audit.variance_status(zero,zero,1.)=='inconclusive'
    assert audit.joint_status(['not_supported','inconclusive'])=='not_supported'
    assert audit.joint_status(['supported','inconclusive'])=='inconclusive'
    assert audit.joint_status(['supported','supported'])=='supported'


def test_method_local_missing_t_and_BCa_without_global_poison():
    panel=fixed_panel(o=1);panel['valid'][0,4,0,119]=False;panel['lower'][0,4,0,119]=panel['upper'][0,4,0,119]=np.nan
    rows=audit.original_metrics(panel);t=next(row for row in rows if row['method']=='t');bca=next(row for row in rows if row['method']=='bca')
    assert not t['complete'] and t['total_inner']==t['invalid_inner']==1 and bca['complete']
    panel['valid'][0,9,2,59]=False;panel['lower'][0,9,2,59]=panel['upper'][0,9,2,59]=np.nan
    row=next(row for row in audit.original_metrics(panel) if row['method']=='bca')
    assert not row['complete'] and row['total_inner']==2400 and row['valid_inner']==2399 and row['d'] is None


def test_input_hash_diagnostic_and_stream_order_tamper_detected():
    panel=fixed_panel();assert audit.inspect_panel(panel,strict_design=False)==[]
    panel['stream_id'][0]=1;assert 'stream IDs differ' in audit.inspect_panel(panel,strict_design=False)
    panel=fixed_panel();panel['sample_sha256'][0]=b'0'*64;assert 'original hash mismatch' in audit.inspect_panel(panel,strict_design=False)
    panel=fixed_panel();panel['tau_low'][0,0,2,0]=1;assert any('tau differs' in message for message in audit.inspect_panel(panel,strict_design=False))


def test_saved_indices_to_means_and_all_diagnostics_without_RNG(tmp_path):
    from mathscape import finite_k
    panel=fixed_panel(o=1);sample=panel['samples'][0]
    indices=((np.arange(4999)[:,None]+np.arange(20)[None,:]//3)%20).astype(np.uint8);means=np.mean(sample[indices],axis=1)
    record=finite_k.infer_single(finite_k.prepare_sample(sample,alpha=.05),means)
    for mi,method in enumerate(audit.METHODS):
        for field in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high','valid','tail_warning'):panel[field][0,0,mi,59]=record[method][field]
    path=tmp_path/'fixed_stream.npz';meta=dict(distribution='gamma',master_seed=0,outer_id=0,budget_repeat_id=0,stream_id=59,n=20,B=4999,R=20,streams_per_budget=120,alpha=.05,namespace=audit.MAIN_NAMESPACE,native_indices_dtype='int64',stored_indices_dtype='uint8',sample_sha256=panel['sample_sha256'][0])
    np.savez_compressed(path,indices=indices,means=means,**meta);assert audit.audit_stream(path,panel,'gamma',0,59)==[]
    means[0]+=.01;np.savez_compressed(path,indices=indices,means=means,**meta)
    assert 'saved means differ from indices/original' in audit.audit_stream(path,panel,'gamma',0,59)
    np.savez_compressed(path,indices=indices.astype(np.int64),means=means,**meta)
    assert 'indices shape/dtype differ' in audit.audit_stream(path,panel,'gamma',0,59)


def test_995_critical_value_original_N_not_groups_R_or_J_and_no_clipping():
    values=np.arange(30,dtype=float)/30;interval=audit.mean_interval(values)
    half=stats.t.ppf(.9975,29)*np.std(values,ddof=1)/np.sqrt(30)
    assert interval['N']==30 and interval['df']==29 and interval['high']==pytest.approx(values.mean()+half)
    assert interval['mcse']==pytest.approx(np.std(values,ddof=1)/np.sqrt(30))
    interval=audit.mean_interval([1,np.nan,3]);assert interval['N']==3 and interval['finite_count']==2 and interval['mean'] is None


def test_every_allocation_uses_all120_once_and_reordering_changes_grouping_T():
    panel=fixed_panel(o=1);raw=panel['lower'][0,0];upper=panel['upper'][0,0];valid=panel['valid'][0,0];warning=panel['tail_warning'][0,0]
    a=audit.budget_metrics(raw,upper,valid,warning,1.)[2]
    for item in a['allocations']:
        assert sum(row['group_stop']-row['group_start'] for row in item['groups'])==120
        assert [(row['group_start'],row['group_stop']) for row in item['groups']]==[(i*item['K'],(i+1)*item['K']) for i in range(item['J'])]
    order=np.arange(120).reshape(4,30).T.ravel();b=audit.budget_metrics(raw[:,order],upper[:,order],valid[:,order],warning[:,order],1.)[2]
    assert a['allocations'][0]['T']!=pytest.approx(b['allocations'][0]['T'])
    assert a['allocations'][2]['T']==pytest.approx(b['allocations'][2]['T'])


def primary_fixture_tables(panel):
    from scripts import run_allocation_benchmark as reference
    output={key:[] for key in ('original_allocations','original_comparisons','budgets','groups','decompositions')}
    for oi,sample in enumerate(panel['samples']):
        raw={key:panel[key][oi] for key in ('lower','upper','valid','tail_warning')};se=float(audit._original_se(sample[None,:])[0])
        result=reference.original_metrics(raw,se)
        tables=reference.derived_tables(result,raw,se,str(panel['distribution']),int(panel['master_seed']),oi,audit.sample_hash(sample),len(panel['samples']),20,120)
        for key,rows in tables.items():output[key].extend(rows)
    return output


@pytest.mark.parametrize('invalid_method',[None,0,1,2])
def test_all_reduction_columns_group_order_counts_NA_against_primary_fixedarrays(invalid_method):
    panel=fixed_panel(o=1)
    if invalid_method is not None:
        panel['valid'][0,7,invalid_method,119]=False;panel['lower'][0,7,invalid_method,119]=panel['upper'][0,7,invalid_method,119]=np.nan
    expected=audit.derived_tables(panel);actual=primary_fixture_tables(panel)
    for key,keys in [('original_allocations',['distribution','master_seed','outer_id','method','allocation']),('original_comparisons',['distribution','master_seed','outer_id','method']),
                     ('budgets',['distribution','master_seed','outer_id','budget_repeat_id','method','allocation']),('groups',['distribution','master_seed','outer_id','budget_repeat_id','method','group_index']),
                     ('decompositions',['distribution','master_seed','outer_id','budget_repeat_id','method'])]:
        assert audit.compare_rows(expected[key],actual[key],keys,label=key)==[]


def test_full_saved_fixture_actual_schema_provenance_all_tables_six_CI_36streams(tmp_path,monkeypatch):
    """Accepted60 originals + fixed identity indices, not a scientific draw."""
    import json
    from scripts import run_allocation_benchmark as reference
    from mathscape import allocation,finite_k,matched_stability,matched_moments
    def forbidden(*args,**kwargs):raise AssertionError('no RNG/main/smoke in deterministic saved fixture')
    for module,names in ((reference,('run_cell','run_benchmark','run_smoke','generate_bootstrap_means')),
                         (allocation,('rng_for_panel','generate_bootstrap_means')),(finite_k,('rng_for_panel','generate_bootstrap_means')),
                         (matched_stability,('rng_for_panel','generate_bootstrap_means')),(matched_moments,('rng_for_panel','generate_sample','generate_bootstrap_means'))):
        for name in names:monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    manifest=reference.make_manifest();inputs=reference.load_inputs(manifest['inputs'],root=reference.ROOT)
    target=tmp_path/'fixed_saved';(target/'panels').mkdir(parents=True);(target/'streams').mkdir()
    tables={key:[] for key in ('original_allocations','original_comparisons','budgets','groups','decompositions')};indices=np.broadcast_to(np.arange(20,dtype=np.uint8),(4999,20))
    for dist,seed in reference.fixed_cells():
        saved=inputs[(dist,seed)];panel=reference._allocate(saved,10,20,120);labels=['']
        for oi,sample in enumerate(saved['samples']):
            context=reference.prepare_sample(sample,alpha=.05);means=np.mean(sample[indices],axis=1);records=reference.infer_single(context,means)
            for mi,method in enumerate(audit.METHODS):
                rec=records[method]
                for field in (*reference.DIAGNOSTICS,'valid','tail_warning'):panel[field][oi,:,mi,:]=rec[field]
                if rec['reason'] not in labels:labels.append(rec['reason'])
                panel['reason_codes'][oi,:,mi,:]=labels.index(rec['reason'])
            raw={key:panel[key][oi] for key in (*reference.DIAGNOSTICS,'valid','tail_warning')};result=reference.original_metrics(raw,float(context['se']))
            local=reference.derived_tables(result,raw,float(context['se']),dist,seed,oi,audit.sample_hash(sample),10,20,120)
            for key,rows in local.items():tables[key].extend(rows)
            if oi==0:
                for stream in (0,29,30,59,60,119):
                    np.savez_compressed(target/'streams'/f'{dist}_seed{seed}_outer0_repeat0_stream{stream}.npz',indices=indices,means=means,distribution=dist,master_seed=seed,outer_id=0,budget_repeat_id=0,stream_id=stream,n=20,B=4999,R=20,streams_per_budget=120,alpha=.05,namespace=audit.MAIN_NAMESPACE,native_indices_dtype='int64',stored_indices_dtype='uint8',sample_sha256=saved['sample_sha256'][0])
        panel['reason_labels']=np.array(labels,dtype='U96');panel['observed_streams'][:]=True;reference.save_panel(target/'panels'/f'{dist}_seed{seed}.npz',panel,dist,seed,20,120,4999)
    for filename,key in [('original_allocation_metrics.csv','original_allocations'),('original_comparison_metrics.csv','original_comparisons')]:reference.write_csv(target/filename,tables[key])
    for filename,key in [('budget_metrics.csv.gz','budgets'),('group_metrics.csv.gz','groups'),('decomposition_metrics.csv.gz','decompositions')]:
        with reference.gzip_table(target/filename) as writer:writer.write_rows(tables[key])
    for filename,rows in [('summary.csv',reference.summarize(tables['original_allocations'])),('pooled_summary.csv',reference.summarize(tables['original_allocations'],True)),('decomposition_summary.csv',reference.decomposition_summary(tables['original_comparisons'],tables['decompositions']))]:reference.write_csv(target/filename,rows)
    reference._write_json(target/'hypotheses.json',reference.assemble_hypotheses(tables['original_comparisons']));reference._write_json(target/'manifest.json',manifest)
    reference._write_json(target/'resources.json',dict(status='completed',completed_cells=6,incomplete_cells=[]))
    result=audit.audit_evidence(target)
    assert result['mismatches']==[]
    assert result['status']=='passed' and result['original_samples']==60 and result['fully_reconstructed_streams']==36
    for name,count in [('original_allocation_metrics.csv',540),('original_comparison_metrics.csv',180),('budget_metrics.csv.gz',10800),('group_metrics.csv.gz',25200),('decomposition_metrics.csv.gz',3600),('summary.csv',54),('pooled_summary.csv',18),('decomposition_summary.csv',24)]:
        assert result['table_counts'][name]==dict(reconstructed=count,saved=count)
    json.dumps(result,allow_nan=False)


def confirmation_rows():
    rows=[]
    for dist in audit.DISTRIBUTIONS:
        for seed in (0,1,2):
            for outer in range(10):
                q30=.001+outer*.00002+seed*.00001;q120=q30*(1+.02*((outer%3)-1))
                if seed==outer==0:q30=q120=0.
                rows.append(dict(distribution=dist,master_seed=seed,outer_id=outer,method='bca',complete=True,d=(outer-4.5)*.00003+(seed-1)*.00001+(0 if dist=='gamma' else .00001),
                                 Q30=q30,Q120=q120,L=q120-.8*q30,U=q120-1.25*q30,ratio_Q120_Q30=q120/q30 if q30>0 else np.nan))
    return rows


def test_six_nonzero_995_intervals_N30_and_zero_individualQ_does_not_poison_H2():
    from scripts import run_allocation_benchmark as reference
    rows=confirmation_rows();independent=audit.hypothesis_reference(rows);primary=reference.assemble_hypotheses(rows)
    assert audit.compare_json(independent,primary)==[] and len(independent['quantities'])==6
    assert all(row['N']==30 and row['finite_count']==30 and row['df']==29 and row['valid'] for row in independent['quantities'].values())
    assert all(row['status']=='supported' for row in independent['hypotheses'].values())
    for dist in audit.DISTRIBUTIONS:
        q=independent['population_decisions'][dist]['H2'];assert q['mean_Q30']>0 and q['status']=='supported'
        values=np.array([row['L'] for row in rows if row['distribution']==dist]);half=stats.t.ppf(.9975,29)*np.std(values,ddof=1)/np.sqrt(30)
        assert independent['quantities']['H2_L_'+dist]['low']==pytest.approx(values.mean()-half)
    assert any(row['Q30']==0 and np.isnan(row['ratio_Q120_Q30']) for row in rows)


def test_missing_necessary_data_keeps_N30_methodlocal_and_clear_otherpopulation_NOT_wins():
    from scripts import run_allocation_benchmark as reference
    rows=[row for row in confirmation_rows() if not(row['distribution']=='gamma' and row['master_seed']==1 and row['outer_id']==3)]
    for row in rows:
        if row['distribution']=='lognormal':row['Q120']=.5*row['Q30'];row['L']=row['Q120']-.8*row['Q30'];row['U']=row['Q120']-1.25*row['Q30']
    result=audit.hypothesis_reference(rows);assert audit.compare_json(result,reference.assemble_hypotheses(rows))==[]
    assert result['quantities']['H2_L_gamma']['N']==30 and result['quantities']['H2_L_gamma']['finite_count']==29 and result['quantities']['H2_L_gamma']['low'] is None
    assert result['population_decisions']['gamma']['H2']['status']=='inconclusive'
    assert result['population_decisions']['lognormal']['H2']['status']=='not_supported' and result['hypotheses']['H2']['status']=='not_supported'


def test_positive_averageQ_gate_not_mean_individual_ratios_and_NA_reference():
    rows=confirmation_rows();result=audit.hypothesis_reference(rows)
    for row in rows:
        if row['distribution']=='gamma':row['Q30']=0.
    held=audit.hypothesis_reference(rows)
    assert held['population_decisions']['gamma']['H2']['status']=='inconclusive' and held['population_decisions']['gamma']['H2']['mean_Q30']==0
    assert result['population_decisions']['gamma']['H2']['status']=='supported'


def test_guard_source_hash_mutation_and_resource_design_changes_are_detectable(tmp_path):
    import hashlib
    guard=tmp_path/'scripts/run_allocation_guarded.py';guard.parent.mkdir();guard.write_text('hard_limit_seconds = 300\n')
    manifest=dict(source_files_sha256={'scripts/run_allocation_guarded.py':hashlib.sha256(guard.read_bytes()).hexdigest()})
    assert audit.check_source_hashes(manifest,root=tmp_path)==[]
    guard.write_text('hard_limit_seconds = 999\n')
    assert any('source hash differs' in error for error in audit.check_source_hashes(manifest,root=tmp_path))
    assert audit.compare_json(dict(wall_seconds=300,max_resample_elements=1000000),dict(wall_seconds=999,max_resample_elements=1000001))


def test_fractional_constant_original_rejected_seNA_no_zero_imputation():
    panel=fixed_panel(o=1);panel['samples'][0]=.8;panel['sample_sha256'][0]=audit.sample_hash(panel['samples'][0]).encode()
    panel['valid'][:]=False;panel['lower'][:]=panel['upper'][:]=np.nan;panel['reason_codes'][:]=1
    rows=audit.original_metrics(panel);assert all(np.isnan(row['se']) and not row['complete'] and row['d'] is None and row['Q30'] is None for row in rows)
    assert rows[0]['total_inner']==rows[0]['invalid_inner']==1


def test_ratio_meanQ_and_mean_individual_ratios_are_different_descriptive_outputs():
    from scripts import run_allocation_benchmark as reference
    tables=primary_fixture_tables(fixed_panel(o=10))
    for row in tables['original_comparisons']:
        if row['method']=='bca':row.update(Q30=float(1+row['outer_id']),Q120=float(.25+.01*row['outer_id']));row['ratio_Q120_Q30']=row['Q120']/row['Q30']
    independent=audit.decomposition_rows(tables['original_comparisons'],tables['decompositions']);primary=reference.decomposition_summary(tables['original_comparisons'],tables['decompositions'])
    assert audit.compare_rows([row for row in independent if row['distribution']=='gamma' and row['master_seed']==0],[row for row in primary if row['distribution']=='gamma' and row['master_seed']==0],['distribution','master_seed','method'])==[]
    row=next(row for row in independent if row['distribution']=='gamma' and row['master_seed']==0 and row['method']=='bca')
    assert row['mean_individual_Q120_Q30']!=pytest.approx(row['ratio_mean_Q120_over_mean_Q30'])
    t=next(row for row in independent if row['distribution']=='gamma' and row['master_seed']==0 and row['method']=='t')
    assert t['mean_individual_Q120_Q30'] is None and t['ratio_mean_Q120_over_mean_Q30'] is None


def test_original_paired_difference_MCSE_differs_from_unpaired_component_formula():
    rows=confirmation_rows()
    for index,row in enumerate(rows):row['M30']=index*.2;row['M120']=row['M30']+row['d']
    result=audit.hypothesis_reference(rows);selected=[row for row in rows if row['distribution']=='gamma']
    paired=np.array([row['M120']-row['M30'] for row in selected]);unpaired=np.sqrt((np.var([row['M120'] for row in selected],ddof=1)+np.var([row['M30'] for row in selected],ddof=1))/30)
    assert result['quantities']['H1_gamma']['mcse']==pytest.approx(np.std(paired,ddof=1)/np.sqrt(30))
    assert result['quantities']['H1_gamma']['mcse']<unpaired/100


def test_t_control_handles_tiny_positive_scale_without_bootstrap_nonfinite_poison():
    raw=np.tile(np.linspace(-1,1,120),(3,1));upper=raw+2;raw[0]=-.5;upper[0]=.5
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        rows=audit.budget_metrics(raw,upper,np.ones_like(raw,bool),np.zeros_like(raw,bool),1e-200)
    assert rows[0]['complete'] and all(a['T']==a['Vbar']==0 for a in rows[0]['allocations'])
    assert all(not row['complete'] and row['reason']=='derived_statistic_nonfinite' for row in rows[1:])
