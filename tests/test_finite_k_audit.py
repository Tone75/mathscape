"""Deterministic finite-K audit references; no random experiment execution."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy import stats
from scripts import audit_finite_k as audit


def fixed_panel(r=2,j=20,k=120):
    samples=np.array([np.linspace(-.5,2,20)+i*.01 for i in range(r)])
    shape=(r,j,3,k)
    panel={key:np.full(shape,np.nan) for key in ("lower","upper","a","z0","q_low","q_high","tau_low","tau_high")}
    panel.update(samples=samples,sample_sha256=np.array([audit.sample_hash(x).encode() for x in samples],dtype="S64"),
                 outer_id=np.arange(r,dtype=np.uint16),replicate_id=np.arange(j,dtype=np.uint8),inner_id=np.arange(k,dtype=np.uint8),
                 methods=np.array(audit.METHODS),observed_streams=np.ones((r,j,k),bool),valid=np.ones(shape,bool),tail_warning=np.zeros(shape,bool),
                 reason_codes=np.zeros(shape,np.uint8),reason_labels=np.array(["","missing_endpoint"],dtype="U96"),
                 distribution=np.array("gamma"),master_seed=np.array(0),n=np.array(20),R=np.array(r),J=np.array(j),K=np.array(k),
                 B=np.array(4999),alpha=np.array(.05),namespace=np.array(audit.MAIN_NAMESPACE),input_accepted_head=np.array(audit.ACCEPTED_INPUT))
    for oi,sample in enumerate(samples):
        half=stats.t.ppf(.975,19)*np.std(sample,ddof=1)/np.sqrt(20)
        panel["lower"][oi,:,0]=sample.mean()-half;panel["upper"][oi,:,0]=sample.mean()+half
        for rep in range(j):
            for mi in (1,2):
                for group in range(4):
                    idx=slice(group*k//4,(group+1)*k//4)
                    scale=.01*(group+1)*(1+.01*rep+.05*oi)
                    pattern=np.linspace(-1,1,k//4)*scale
                    panel["lower"][oi,rep,mi,idx]=-.4+pattern
                    panel["upper"][oi,rep,mi,idx]=.5+1.2*pattern
                panel["q_low"][oi,rep,mi]=.025;panel["q_high"][oi,rep,mi]=.975
                panel["tau_low"][oi,rep,mi]=panel["tau_high"][oi,rep,mi]=4999*.025
                panel["a"][oi,rep,mi]=0
                if mi==2:panel["z0"][oi,rep,mi]=0
    return panel


def test_auditor_imports_no_primary_models_aggregators_or_old_auditors():
    tree=ast.parse(Path(audit.__file__).read_text())
    names=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any(name and ("mathscape" in name or "run_finite" in name or "audit_matched" in name) for name in names)


def test_predeclared_numeric_and_identity_tolerances_are_fixed():
    assert (audit.NUMERIC_RTOL,audit.NUMERIC_ATOL)==(2e-10,2e-12)
    assert (audit.IDENTITY_RTOL,audit.IDENTITY_ATOL)==(2e-12,2e-12)


def test_hand_calculated_group_variances_concavity_pooling_and_delta_identity():
    lo=np.tile(np.array([-1,1,-2,2,-3,3,-4,4.]),(3,1));hi=lo+2
    lo[0]=-.5;hi[0]=.5
    rows=audit.replicate_metrics(lo,hi,np.ones_like(lo,bool),np.zeros_like(lo,bool),se=1.)
    row=rows[2]
    np.testing.assert_allclose(row["group_V"],[2,8,18,32])
    assert row["V30"]==pytest.approx(2) and row["V120"]==pytest.approx(60/7)
    assert row["Vbar"]==pytest.approx(15) and row["Abar"]==pytest.approx(2.5*np.sqrt(2))
    assert row["Jgap"]==pytest.approx(np.sqrt(15)-2.5*np.sqrt(2))
    assert row["Jgap"]>0 and row["Pgap"]<0 and row["Rgap"]>0
    assert row["delta"]==pytest.approx(row["Jgap"]+row["Pgap"]+row["Rgap"])
    assert abs(row["lower_ss_residual"])<=audit.IDENTITY_ATOL+audit.IDENTITY_RTOL*row["lower_ss_scale"]
    assert row["A30"]!=row["Abar"]  # first group is primary; no four-group substitution
    assert row["algebra_ok"] and row["complete"]
    assert rows[0]["V30"]==rows[0]["V120"]==rows[0]["Jgap"]==rows[0]["delta"]==0


def test_original_repeat_variance_d_W_and_ratios_are_separate_quantities():
    rows=audit.original_metrics(fixed_panel(r=1))
    row=next(row for row in rows if row["method"]=="bca")
    first=np.array([rep["A30"] for rep in row["replicates"]]);full=np.array([rep["A120"] for rep in row["replicates"]])
    assert row["complete"] and row["planned_replicates"]==20
    assert row["d"]==pytest.approx(np.mean(full-first))
    assert row["Q30"]==pytest.approx(np.var(first,ddof=1))
    assert row["Q120"]==pytest.approx(np.var(full,ddof=1))
    assert row["W"]==pytest.approx(row["Q120"]-.5*row["Q30"])
    assert row["ratio_Q120_Q30"]==pytest.approx(row["Q120"]/row["Q30"])
    t=next(row for row in rows if row["method"]=="t")
    assert t["d"]==t["Q30"]==t["Q120"]==t["W"]==0 and t["ratio_Q120_Q30"] is None


def test_one_missing_t_does_not_raise_or_poison_complete_BCa():
    panel=fixed_panel(r=1)
    panel["valid"][0,7,0,8]=False
    panel["reason_codes"][0,7,0,8]=1
    panel["lower"][0,7,0,8]=panel["upper"][0,7,0,8]=np.nan
    assert audit.inspect_panel(panel,strict_design=False)==[]
    rows=audit.original_metrics(panel)
    t=next(row for row in rows if row["method"]=="t");bca=next(row for row in rows if row["method"]=="bca")
    assert not t["complete"] and t["total_inner"]==t["invalid_inner"]==1 and t["valid_inner"]==0
    assert t["d"] is None and bca["complete"]
    panel["lower"][0,1,0,0]+=.01
    with pytest.raises(ValueError,match="observed t endpoints vary"):
        audit.original_metrics(panel)


def test_one_invalid_BCa_K_member_keeps_full_counts_and_NA():
    panel=fixed_panel(r=1)
    panel["valid"][0,4,2,119]=False
    panel["lower"][0,4,2,119]=panel["upper"][0,4,2,119]=np.nan
    rows=audit.original_metrics(panel)
    bca=next(row for row in rows if row["method"]=="bca")
    assert not bca["complete"] and bca["total_inner"]==20*120
    assert bca["invalid_inner"]==1 and bca["valid_inner"]==2399
    assert bca["complete_replicates"]==19 and bca["d"] is None and bca["Q120"] is None
    assert next(row for row in rows if row["method"]=="percentile")["complete"]


@pytest.mark.parametrize("lo,hi,defined,complete,status",[
    (-.0024,.0024,True,True,"supported"),(-.0025,.0024,True,True,"inconclusive"),
    (-.0024,.0025,True,True,"inconclusive"),(.0026,.004,True,True,"not_supported"),
    (-.004,-.0026,True,True,"not_supported"),(0,0,False,True,"inconclusive"),
    (-.0024,.0024,True,False,"inconclusive")])
def test_H1_band_equivalence_is_supported_with_strict_Ase_margin(lo,hi,defined,complete,status):
    assert audit.band_status(dict(low=lo,high=hi,defined=defined),complete)==status


@pytest.mark.parametrize("lo,hi,defined,status",[(-.2,-.1,True,"supported"),(0,.1,True,"not_supported"),
    (-.1,0,True,"inconclusive"),(0,0,False,"inconclusive")])
def test_H2_variance_reduction_low_zero_and_SE0(lo,hi,defined,status):
    assert audit.reduction_status(dict(low=lo,high=hi,defined=defined))==status


@pytest.mark.parametrize("statuses,status",[(["supported","supported"],"supported"),
    (["supported","inconclusive"],"inconclusive"),(["not_supported","inconclusive"],"not_supported")])
def test_joint_status_never_discards_a_clear_counterexample(statuses,status):
    assert audit.joint_status(statuses)==status


def test_endpoint_inspection_detects_hash_q_tau_and_tampered_ids():
    panel=fixed_panel()
    assert audit.inspect_panel(panel,strict_design=False)==[]
    panel["tau_low"][0,0,2,0]=99
    assert any("tau differs" in message for message in audit.inspect_panel(panel,strict_design=False))
    panel["sample_sha256"][0]=b"0"*64
    assert any("hash mismatch" in message for message in audit.inspect_panel(panel,strict_design=False))


def test_full_saved_schema_fixture_all_tables_provenance_streams_and_confirmation(tmp_path,monkeypatch):
    """Reuse accepted originals with fixed identity indices; never draw RNG."""
    import json
    from scripts import run_finite_k_benchmark as reference
    from mathscape import finite_k,matched_stability,matched_moments
    def forbidden(*args,**kwargs):raise AssertionError('no generation or experiment in audit fixture')
    for module,names in ((reference,('run_benchmark','run_smoke','run_cell','generate_bootstrap_means')),
                         (finite_k,('generate_bootstrap_means','rng_for_panel')),
                         (matched_stability,('generate_bootstrap_means','rng_for_panel')),
                         (matched_moments,('generate_sample','generate_bootstrap_means','rng_for_panel'))):
        for name in names:
            if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    manifest=reference.make_manifest();inputs=reference.load_inputs(manifest['inputs'],root=reference.ROOT)
    target=tmp_path/'fixed_saved';(target/'panels').mkdir(parents=True);(target/'streams').mkdir()
    all_tables={name:[] for name in ('originals','original_K','replicates','groups','K_rows')}
    indices=np.broadcast_to(np.arange(20,dtype=np.uint8),(4999,20))
    for dist,seed in reference.fixed_cells():
        saved=inputs[(dist,seed)];panel=reference._allocate(saved,10,20,120);labels=['']
        for oi,sample in enumerate(saved['samples']):
            context=reference.prepare_sample(sample,alpha=.05);means=np.mean(sample[indices],axis=1)
            records=reference.infer_single(context,means)
            for mi,method in enumerate(audit.METHODS):
                rec=records[method]
                for key in (*reference.DIAGNOSTICS,'valid','tail_warning'):panel[key][oi,:,mi,:]=rec[key]
                if rec['reason'] not in labels:labels.append(rec['reason'])
                panel['reason_codes'][oi,:,mi,:]=labels.index(rec['reason'])
            raw={key:panel[key][oi] for key in (*reference.DIAGNOSTICS,'valid','tail_warning')}
            result=reference.original_metrics(raw,float(context['se']))
            tables=reference.derived_tables(result,raw,float(context['se']),dist,seed,oi,audit.sample_hash(sample),10,20,120)
            for key,rows in tables.items():all_tables[key].extend(rows)
            if oi==0:
                for inner in (0,29,30,119):
                    np.savez_compressed(target/'streams'/f'{dist}_seed{seed}_outer0_replicate0_inner{inner}.npz',indices=indices,means=means,
                        distribution=dist,master_seed=seed,outer_id=0,replicate_id=0,inner_id=inner,n=20,B=4999,J=20,K=120,alpha=.05,namespace=audit.MAIN_NAMESPACE,
                        native_indices_dtype='int64',stored_indices_dtype='uint8',sample_sha256=saved['sample_sha256'][0])
        panel['observed_streams'][:]=True
        panel['reason_labels']=np.array(labels,dtype='U96');reference.save_panel(target/'panels'/f'{dist}_seed{seed}.npz',panel,dist,seed,20,120,4999)
    for filename,name in (('original_metrics.csv','originals'),('original_K_metrics.csv','original_K')):reference.write_csv(target/filename,all_tables[name])
    for filename,name in (('replicate_metrics.csv.gz','replicates'),('group_metrics.csv.gz','groups'),('K_metrics.csv.gz','K_rows')):
        with reference.gzip_table(target/filename) as writer:writer.write_rows(all_tables[name])
    for filename,rows in (('summary.csv',reference.summarize(all_tables['original_K'])),('pooled_summary.csv',reference.summarize(all_tables['original_K'],True)),
                          ('decomposition_summary.csv',reference.decomposition_summary(all_tables['originals'],all_tables['replicates']))):reference.write_csv(target/filename,rows)
    reference._write_json(target/'hypotheses.json',reference.assemble_hypotheses(all_tables['originals']))
    reference._write_json(target/'manifest.json',manifest);reference._write_json(target/'resources.json',dict(status='completed',completed_cells=6,incomplete_cells=[]))
    result=audit.audit_evidence(target)
    assert result['mismatches']==[]
    assert result['status']=='passed' and result['original_samples']==60 and result['fully_reconstructed_streams']==24
    for name,count in [('original_metrics.csv',180),('original_K_metrics.csv',360),('replicate_metrics.csv.gz',3600),('group_metrics.csv.gz',14400),('K_metrics.csv.gz',7200),('summary.csv',36),('pooled_summary.csv',12),('decomposition_summary.csv',24)]:
        assert result['table_counts'][name]==dict(reconstructed=count,saved=count)
    json.dumps(result,allow_nan=False)


def primary_fixture_tables(panel):
    from scripts import run_finite_k_benchmark as reference
    all_tables={name:[] for name in ('originals','original_K','replicates','groups','K_rows')}
    for oi,sample in enumerate(panel['samples']):
        raw={key:panel[key][oi] for key in ('lower','upper','valid','tail_warning')}
        se=float(audit._original_se(sample[None,:])[0]);metrics=reference.original_metrics(raw,se)
        tables=reference.derived_tables(metrics,raw,se,str(panel['distribution']),int(panel['master_seed']),oi,audit.sample_hash(sample),len(panel['samples']),20,120)
        for key,rows in tables.items():all_tables[key].extend(rows)
    return all_tables


@pytest.mark.parametrize('invalid_method',[None,0,1,2])
def test_independent_all_derived_scalar_group_K_fields_against_primary_fixed_arrays(invalid_method):
    panel=fixed_panel(r=1)
    if invalid_method is not None:
        panel['valid'][0,4,invalid_method,119]=False
        panel['lower'][0,4,invalid_method,119]=panel['upper'][0,4,invalid_method,119]=np.nan
    primary=primary_fixture_tables(panel);independent=audit.derived_tables(panel)
    for key,keys in [('originals',['distribution','master_seed','outer_id','method']),('original_K',['distribution','master_seed','outer_id','method','K']),
                      ('replicates',['distribution','master_seed','outer_id','replicate_id','method']),('groups',['distribution','master_seed','outer_id','replicate_id','method','group']),
                      ('K_rows',['distribution','master_seed','outer_id','replicate_id','method','K'])]:
        assert audit.compare_rows(independent[key],primary[key],keys,label=key)==[]


def test_missing_original_retains_planned_N30_and_local_target_NA():
    from scripts import run_finite_k_benchmark as reference
    all_tables={name:[] for name in ('originals','original_K','replicates','groups','K_rows')}
    for dist in audit.DISTRIBUTIONS:
        for seed in (0,1,2):
            panel=fixed_panel(r=10);panel['distribution']=np.array(dist);panel['master_seed']=np.array(seed)
            tables=primary_fixture_tables(panel)
            for key,rows in tables.items():all_tables[key].extend(rows)
    independent=audit.summary_rows(all_tables['original_K'],True)
    assert audit.compare_rows(independent,reference.summarize(all_tables['original_K'],True),['distribution','master_seed','method','K'])==[]
    original=all_tables['originals'];original=[r for r in original if not(r['distribution']=='gamma' and r['master_seed']==1 and r['outer_id']==5 and r['method']=='bca')]
    actual=audit.hypothesis_reference(original);expected=reference.assemble_hypotheses(original)
    assert audit.compare_json(actual,expected)==[]
    assert actual['quantities']['H1_gamma']['N']==30 and actual['quantities']['H1_gamma']['finite_count']==29
    assert actual['quantities']['H1_gamma']['status']=='inconclusive' and actual['quantities']['H1_gamma']['low'] is None
    assert actual['quantities']['H1_lognormal']['finite_count']==30


def test_designated_indices_reconstruct_means_half_ties_BCa_and_detect_tamper(tmp_path):
    from mathscape import finite_k
    panel=fixed_panel(r=1);sample=panel['samples'][0]
    indices=((np.arange(4999)[:,None]+np.arange(20)[None,:]//3)%20).astype(np.uint8)
    means=np.mean(sample[indices],axis=1)
    records=finite_k.infer_single(finite_k.prepare_sample(sample,alpha=.05),means)
    for mi,method in enumerate(audit.METHODS):
        for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high','valid','tail_warning'):
            panel[key][0,0,mi,29]=records[method][key]
    path=tmp_path/'stream.npz'
    metadata=dict(distribution='gamma',master_seed=0,outer_id=0,replicate_id=0,inner_id=29,n=20,B=4999,J=20,K=120,alpha=.05,namespace=audit.MAIN_NAMESPACE,
                  native_indices_dtype='int64',stored_indices_dtype='uint8',sample_sha256=panel['sample_sha256'][0])
    np.savez_compressed(path,indices=indices,means=means,**metadata)
    assert audit.audit_stream(path,panel,'gamma',0,29)==[]
    means[0]+=.01
    np.savez_compressed(path,indices=indices,means=means,**metadata)
    assert 'saved means differ from indices/original' in audit.audit_stream(path,panel,'gamma',0,29)
    np.savez_compressed(path,indices=indices.astype(np.int64),means=means,**metadata)
    assert 'indices shape/dtype differ' in audit.audit_stream(path,panel,'gamma',0,29)


def test_fractional_constant_original_rejected_without_invented_se():
    panel=fixed_panel(r=1);panel['samples'][0]=.8;panel['sample_sha256'][0]=audit.sample_hash(panel['samples'][0]).encode()
    panel['valid'][:]=False;panel['lower'][:]=panel['upper'][:]=np.nan;panel['reason_codes'][:]=1
    rows=audit.original_metrics(panel)
    assert all(not row['complete'] and np.isnan(row['se']) and row['d'] is None and row['Q30'] is None for row in rows)
    assert rows[0]['total_inner']==rows[0]['invalid_inner']==1
    assert rows[1]['total_inner']==20*120


def test_linear_quantiles_original_MCSE_and_t99_interval():
    values=np.array([1.,2.,4.])
    result=audit.metric_statistics(values)
    assert result['p05']==pytest.approx(1.1) and result['p50']==2 and result['p95']==pytest.approx(3.8)
    assert result['mcse']==pytest.approx(np.std(values,ddof=1)/np.sqrt(3))
    assert result['decision_high']==pytest.approx(values.mean()+stats.t.ppf(.995,2)*result['mcse'])
    result=audit.metric_statistics([1,np.nan,4]);assert result['mean'] is None and result['p05'] is None
    result=audit.metric_statistics([.8]*30);assert result['mean']==pytest.approx(.8) and result['sd']==result['mcse']==0 and result['decision_low'] is None


def test_decomposition_raw_signs_are_descriptive_original_N_is_not_J_times_N():
    panel=fixed_panel(r=10);tables=audit.derived_tables(panel)
    rows=audit.decomposition_rows(tables['originals'],tables['replicates'])
    selected=next(row for row in rows if row['distribution']=='gamma' and row['master_seed']==0 and row['method']=='bca')
    assert selected['diagnostic_replicates']==200 and selected['original_total']==10
    assert selected['Jgap_positive_count']==200 and selected['Jgap_original_N']==10
    assert selected['Jgap_original_mcse']>0


def test_mean_individual_Q_ratio_and_ratio_of_mean_Q_are_named_distinctly():
    from scripts import run_finite_k_benchmark as reference
    tables=primary_fixture_tables(fixed_panel(r=10))
    for row in tables['originals']:
        if row['method']=='bca':
            row['Q30']=float(1+row['outer_id']);row['Q120']=float(.25+.01*row['outer_id'])
            row['ratio_Q120_Q30']=row['Q120']/row['Q30']
    independent=audit.decomposition_rows(tables['originals'],tables['replicates']);primary=reference.decomposition_summary(tables['originals'],tables['replicates'])
    assert audit.compare_rows(independent,primary,['distribution','master_seed','method'])==[]
    row=next(row for row in independent if row['distribution']=='gamma' and row['master_seed']==0 and row['method']=='bca')
    assert row['ratio_planned_originals']==row['ratio_finite_originals']==10
    assert row['mean_individual_Q120_Q30']!=pytest.approx(row['ratio_mean_Q120_over_mean_Q30'])
    row=next(row for row in independent if row['distribution']=='gamma' and row['master_seed']==0 and row['method']=='t')
    assert row['mean_individual_Q120_Q30'] is None and row['ratio_mean_Q120_over_mean_Q30'] is None
    assert row['ratio_missing_or_undefined_originals']==10


def test_missing_required_observation_mask_is_not_a_scientific_invalid_endpoint():
    panel=fixed_panel();panel['observed_streams'][0,3,19]=False
    assert 'uncomputed required streams remain' in audit.inspect_panel(panel,strict_design=False)


def test_all_comparison_fields_counts_NA_and_raw_sign_tampering_detected():
    expected=[dict(id=0,raw=1e-4,mcse=2e-6,invalid=0,low=None,sign=-1,complete=True)]
    assert audit.compare_rows(expected,[dict(expected[0])],['id'])==[]
    for field,value in [('raw',.2),('mcse',.01),('invalid',1),('low',0),('sign',1),('complete',False)]:
        observed=dict(expected[0]);observed[field]=value
        assert any(field in error for error in audit.compare_rows(expected,[observed],['id']))
