"""Deterministic n5 saved-audit contracts; no M10 scientific generation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import special,stats
from scripts import audit_small_sample as audit


def original_cell(dist='gamma',seed=0):
    raw=np.array([np.linspace(.1,2.3,5)+(seed*10+outer)*.001 if dist=='gamma' else np.linspace(-1.7,1.9,5)+(seed*10+outer)*.003 for outer in range(10)],dtype='<f8')
    samples=audit.transform_raw(raw,dist);constants=audit.analytic_constants(dist)
    return dict(raw=raw,samples=samples,raw_sha256=np.array([audit.sample_hash(row).encode() for row in raw],dtype='S64'),sample_sha256=np.array([audit.sample_hash(row).encode() for row in samples],dtype='S64'),outer_id=np.arange(10,dtype=np.uint16),original_keys=np.array([audit.expected_original_key(dist,outer) for outer in range(10)],dtype=np.uint32),
                distribution=np.array(dist),master_seed=np.array(seed),n=np.array(5),namespace=np.array(audit.MAIN_NAMESPACE),raw_kind=np.array('gamma_y' if dist=='gamma' else 'normal_z'),constant_names=np.array(list(constants)),constant_values=np.array(list(constants.values()),dtype='<f8'))


def fixed_panel(o=2,r=20,k=120):
    cell=original_cell();samples=cell['samples'][:o].copy();shape=(o,r,3,k)
    panel={key:np.full(shape,np.nan) for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high')}
    panel.update(samples=samples,sample_sha256=cell['sample_sha256'][:o].copy(),outer_id=np.arange(o,dtype=np.uint16),budget_repeat_id=np.arange(r,dtype=np.uint8),stream_id=np.arange(k,dtype=np.uint8),methods=np.array(audit.METHODS),valid=np.ones(shape,bool),tail_warning=np.zeros(shape,bool),reason_codes=np.zeros(shape,np.uint8),reason_labels=np.array(['','endpoint_invalid','not_observed'],dtype='U96'),observed_streams=np.ones((o,r,k),bool),
                 less_count=np.full(shape,2499,np.uint16),equal_count=np.ones(shape,np.uint16),distribution=np.array('gamma'),master_seed=np.array(0),n=np.array(5),O=np.array(o),R=np.array(r),streams_per_budget=np.array(k),B=np.array(4999),alpha=np.array(.05),namespace=np.array(audit.MAIN_NAMESPACE),reference_accepted_head=np.array(audit.ACCEPTED_INPUT),cohort=np.array('M10'))
    for oi,sample in enumerate(samples):
        se=np.std(sample,ddof=1)/np.sqrt(5);half=stats.t.ppf(.975,4)*se;a=audit.jackknife_acceleration(sample)
        panel['lower'][oi,:,0]=sample.mean()-half;panel['upper'][oi,:,0]=sample.mean()+half;panel['a'][oi]=a
        z=special.ndtri(.025);qs=special.ndtr(np.array([z,-z])/(1-a*np.array([z,-z])))
        for rep in range(r):
            for mi in (1,2):
                for group in range(4):
                    sl=slice(group*k//4,(group+1)*k//4);pattern=np.linspace(-1,1,k//4)*.01*(group+1)*(1+rep*.01+oi*.05)
                    panel['lower'][oi,rep,mi,sl]=-.4+pattern;panel['upper'][oi,rep,mi,sl]=.5+pattern*1.2
                lo,hi=(.025,.975) if mi==1 else qs
                panel['q_low'][oi,rep,mi]=lo;panel['q_high'][oi,rep,mi]=hi;panel['tau_low'][oi,rep,mi]=4999*min(lo,1-lo);panel['tau_high'][oi,rep,mi]=4999*min(hi,1-hi)
                if mi==2:panel['z0'][oi,rep,mi]=0
    return panel


def test_no_primary_aggregator_decision_previous_auditor_imports_and_fixed_tolerances():
    tree=ast.parse(Path(audit.__file__).read_text());names=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any(name and ('mathscape' in name or 'run_small_sample' in name or 'audit_replication' in name) for name in names)
    assert (audit.N,audit.MAIN_NAMESPACE)==(5,0x4D3130)
    assert (audit.NUMERIC_RTOL,audit.NUMERIC_ATOL,audit.IDENTITY_RTOL,audit.IDENTITY_ATOL)==(2e-10,2e-12,2e-12,2e-12)


@pytest.mark.parametrize('dist',['gamma','lognormal'])
def test_raw_n5_transform_exact_unrounded_analytic_constants_no_sample_standardization(dist):
    from mathscape import matched_moments
    cell=original_cell(dist);assert audit.inspect_original_cell(cell,dist,0)==[]
    assert cell['samples'].shape==cell['raw'].shape==(10,5)
    assert cell['constant_values'].tobytes()==np.array(list(matched_moments.population_info(dist)['constants'].values()),dtype='<f8').tobytes()
    assert not np.allclose(cell['samples'].mean(axis=1),0) and not np.allclose(cell['samples'].std(axis=1,ddof=1),1)


def test_n5_t_df4_SEsqrt5_Apop_and_sample_comparison_df29_are_distinct():
    sample=np.array([-1.,-.5,0.,.5,3.]);means=np.linspace(-2,3,4999);records=audit.reference_intervals(sample,means);se=np.std(sample,ddof=1)/np.sqrt(5)
    np.testing.assert_allclose([records['t']['lower'],records['t']['upper']],sample.mean()+np.array([-1,1])*stats.t.ppf(.975,4)*se,rtol=1e-13)
    row=audit.budget_metrics(fixed_panel(o=1)['lower'][0,0],fixed_panel(o=1)['upper'][0,0],np.ones((3,120),bool),np.zeros((3,120),bool),se)[2]['allocations'][0]
    assert row['Apop']==pytest.approx(row['S']*np.sqrt(5)) and row['Apop']!=pytest.approx(row['S']*np.sqrt(20))
    interval=audit.mean_interval(np.arange(30.)/30);assert interval['N']==30 and interval['df']==29
    assert interval['high']==pytest.approx(np.mean(np.arange(30.)/30)+stats.t.ppf(.9975,29)*np.std(np.arange(30.)/30,ddof=1)/np.sqrt(30))


def test_n5_key_lengths_namespace_and_5element_shape_not_old20():
    assert audit.expected_original_key('gamma',2)==(0x4D3130,1,5,2,0,0)
    assert audit.expected_inner_key('lognormal',3,19,119)==(0x4D3130,2,5,3,1,19,119)
    bad=original_cell();bad['samples']=np.tile(bad['samples'],(1,4));bad['raw']=np.tile(bad['raw'],(1,4));assert audit.inspect_original_cell(bad,'gamma',0)


def test_delete_one_acceleration_scaled_without_underflow_and_half_ties_linear_quantile():
    from mathscape import small_sample
    sample=np.array([-1.,-.2,0.,.4,2.]);mean=sample.mean();means=np.r_[np.full(1999,mean-1),np.full(1001,mean),np.full(1999,mean+2)]
    reference=audit.reference_intervals(sample,means);primary=small_sample.infer_single(small_sample.prepare_sample(sample),means)
    for method in audit.METHODS:assert audit.compare_rows([dict(id=0,**reference[method])],[dict(id=0,**primary[method])],['id'])==[]
    assert reference['bca']['less_count']==1999 and reference['bca']['equal_count']==1001
    assert reference['bca']['z0']==special.ndtri((1999+.5*1001)/4999)
    jack=(sample.sum()-sample)/4;centered=jack.mean()-jack
    assert audit.jackknife_acceleration(sample)==pytest.approx(np.sum(centered**3)/(6*np.sum(centered**2)**1.5))
    tiny=sample*1e-100;assert audit.jackknife_acceleration(tiny)==pytest.approx(audit.jackknife_acceleration(sample))
    np.testing.assert_array_equal([reference['percentile']['lower'],reference['percentile']['upper']],np.quantile(means,[.025,.975],method='linear'))


@pytest.mark.parametrize('sign',[1,-1])
def test_valid_q_endpoint_warning_is_not_invalid_or_new_gate(sign):
    from mathscape import small_sample
    offset=0 if sign==1 else 4
    sample=sign*np.array([-1.,-1.,-1.,-1.,4.])+offset;means=sign*np.r_[np.full(4998,-1.),4.]+offset
    actual=audit.reference_intervals(sample,means);primary=small_sample.infer_single(small_sample.prepare_sample(sample),means)
    for method in audit.METHODS:assert audit.compare_rows([dict(id=0,**actual[method])],[dict(id=0,**primary[method])],['id'])==[]
    bound,tail=('q_high','tau_high') if sign==1 else ('q_low','tau_low')
    assert actual['bca']['valid'] and actual['bca']['tail_warning']
    if sign==1:assert actual['bca'][bound]==1 and actual['bca'][tail]==0
    else:assert 0<actual['bca'][bound]<1e-60 and 0<actual['bca'][tail]<10  # tiny finite q is not clipped to0
    assert actual['bca']['reason']==''


@pytest.mark.parametrize('sample,means,reason',[ (np.full(5,.8),np.arange(4999.),'sample_constant'),(np.array([-1.,0.,1.,2.,3.]),np.full(4999,-10.),'bca_bias_correction_nonfinite'),(np.array([-1.,0.,1.,2.,3.]),np.full(4999,np.nan),'bootstrap_distribution_nonfinite')])
def test_true_invalid_boundaries_reasons_and_unknown_counts_are_preserved(sample,means,reason):
    from mathscape import small_sample
    reference=audit.reference_intervals(sample,means);primary=small_sample.infer_single(small_sample.prepare_sample(sample),means)
    for method in audit.METHODS:assert audit.compare_rows([dict(id=0,**reference[method])],[dict(id=0,**primary[method])],['id'])==[]
    assert not reference['bca']['valid'] and reference['bca']['reason']==reason
    assert np.isnan(reference['bca']['lower']) and np.isnan(reference['bca']['upper'])
    if reason!='bca_bias_correction_nonfinite':assert reference['t']['less_count']==reference['t']['equal_count']==65535
    else:assert reference['t']['valid'] and reference['percentile']['valid'] and np.isposinf(reference['bca']['z0'])


def test_inspect_panel_counts_z0_actual_half_ties_dtypes_and_method_sharing():
    panel=fixed_panel();assert audit.inspect_panel(panel,strict_design=False)==[]
    panel['equal_count'][0,0,2,0]+=2;assert any('share' in error or 'half-tie' in error for error in audit.inspect_panel(panel,strict_design=False))
    panel=fixed_panel();panel['less_count'][0,0,:,0]=5000;assert 'tie count partition exceedsB' in audit.inspect_panel(panel,strict_design=False)
    panel=fixed_panel();panel['less_count'][0,0,:,0]=65535;assert 'partial unknown tie count' in audit.inspect_panel(panel,strict_design=False)


def test_nonfinite_signs_not_silently_equated_to_NA():
    assert audit.compare_rows([dict(id=0,z=-np.inf)],[dict(id=0,z='-inf')],['id'])==[]
    assert audit.compare_rows([dict(id=0,z=-np.inf)],[dict(id=0,z='inf')],['id'])
    assert audit.compare_rows([dict(id=0,z=-np.inf)],[dict(id=0,z='nan')],['id'])
    assert audit.compare_rows([dict(id=0,z=None)],[dict(id=0,z='inf')],['id'])


def test_T_mean_sqrt_groups_Vbar_not_Tsquare_both_SS_no_Rgap():
    lower=np.tile([-1.,1.,-2.,2.,-3.,3.,-4.,4.],(3,1));upper=lower+2;lower[0]=-.5;upper[0]=.5
    row=audit.budget_metrics(lower,upper,np.ones_like(lower,bool),np.zeros_like(lower,bool),1.)[2]
    assert row['allocations'][0]['T']==pytest.approx(2.5*np.sqrt(2)) and row['allocations'][0]['Vbar']==pytest.approx(15)
    assert row['allocations'][0]['Vse']!=pytest.approx(row['allocations'][0]['T']**2)
    diagnostics=row['diagnostics'];assert diagnostics['delta']==pytest.approx(diagnostics['Jgap']+diagnostics['Pgap']) and not any('Rgap' in field for field in diagnostics)
    for part in (30,60):
        for side in ('lower','upper'):assert abs(diagnostics[f'ss{part}_{side}_residual'])<=audit.IDENTITY_ATOL+audit.IDENTITY_RTOL*diagnostics[f'ss{part}_{side}_scale']


def primary_tables(panel):
    from scripts import run_small_sample_benchmark as reference
    output={key:[] for key in ('original_allocations','original_comparisons','budgets','groups','decompositions')}
    for oi,sample in enumerate(panel['samples']):
        raw={key:panel[key][oi] for key in ('lower','upper','valid','tail_warning')};se=float(audit._original_se(sample[None,:])[0]);result=reference.original_metrics(raw,se)
        local=reference.derived_tables(result,raw,se,str(panel['distribution']),int(panel['master_seed']),int(panel['outer_id'][oi]),audit.sample_hash(sample),len(panel['samples']),20,120)
        for key,rows in local.items():output[key].extend(rows)
    return output


@pytest.mark.parametrize('invalid_method',[None,0,1,2])
def test_all_n5_reduction_columns_and_method_local_invalid_propagation(invalid_method):
    panel=fixed_panel(o=1)
    if invalid_method is not None:panel['valid'][0,9,invalid_method,119]=False;panel['lower'][0,9,invalid_method,119]=panel['upper'][0,9,invalid_method,119]=np.nan
    expected=audit.derived_tables(panel);observed=primary_tables(panel)
    for key,keys in [('original_allocations',['distribution','master_seed','outer_id','method','allocation']),('original_comparisons',['distribution','master_seed','outer_id','method']),('budgets',['distribution','master_seed','outer_id','budget_repeat_id','method','allocation']),('groups',['distribution','master_seed','outer_id','budget_repeat_id','method','group_index']),('decompositions',['distribution','master_seed','outer_id','budget_repeat_id','method'])]:assert audit.compare_rows(expected[key],observed[key],keys,label=key)==[]


def test_full_actual_saved_schema_n5_with_true_invalid_warning_ties_no_RNG(tmp_path,monkeypatch):
    """Fixed raws/indices, includes expected mathematical invalidity, never main."""
    from scripts import run_small_sample_benchmark as reference
    from mathscape import small_sample,replication,allocation,finite_k,matched_stability,matched_moments
    def forbidden(*args,**kwargs):raise AssertionError('no M10 draw, PRNG, main or smoke in fixed schema fixture')
    for module,names in ((reference,('run_cell','run_benchmark','run_smoke','draw_original','generate_cohort','generate_bootstrap_means')),(small_sample,('rng_for_original','rng_for_panel','draw_original','generate_cohort','generate_bootstrap_means')),(replication,('rng_for_original','rng_for_panel','draw_original','generate_cohort','generate_bootstrap_means')),(allocation,('rng_for_panel','generate_bootstrap_means')),(finite_k,('rng_for_panel','generate_bootstrap_means')),(matched_stability,('rng_for_panel','generate_bootstrap_means')),(matched_moments,('rng_for_panel','generate_sample','generate_bootstrap_means'))):
        for name in names:monkeypatch.setattr(module,name,forbidden)
    for name in ('default_rng','Generator','PCG64','SeedSequence'):monkeypatch.setattr(np.random,name,forbidden)
    manifest=reference.make_manifest();loaded_reference=reference.load_reference(manifest['reference'],root=reference.ROOT);target=tmp_path/'fixed_n5'
    (target/'panels').mkdir(parents=True);(target/'streams').mkdir();reference._write_json(target/'manifest.json',manifest);manifest['_manifest_sha256']=hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest()
    source_commit='f'*40;actual_run=audit.subprocess.run
    def fixture_git(args,**kwargs):
        if args[:2]==['git','show'] and args[2].startswith(source_commit+':'):
            name=args[2].split(':',1)[1];data=(target/'manifest.json').read_bytes() if name=='evidence/rq_m10/manifest.json' else (reference.ROOT/name).read_bytes()
            return audit.subprocess.CompletedProcess(args,0,stdout=data,stderr=b'')
        return actual_run(args,**kwargs)
    monkeypatch.setattr(audit.subprocess,'run',fixture_git)
    cells={(dist,seed):original_cell(dist,seed) for dist,seed in reference.fixed_cells()}
    special_cell=cells[('gamma',0)];special_cell['raw'][0]=[0,0,0,0,5];special_cell['raw'][2]=[1.8]*5;special_cell['samples']=audit.transform_raw(special_cell['raw'],'gamma')
    special_cell['sample_sha256']=np.array([audit.sample_hash(row).encode() for row in special_cell['samples']],dtype='S64');special_cell['raw_sha256']=np.array([audit.sample_hash(row).encode() for row in special_cell['raw']],dtype='S64')
    reference.persist_cohort(target/'originals',cells,loaded_reference,source_commit,manifest);cells,receipt=reference.readback_cohort(target/'originals',loaded_reference,manifest,source_commit)
    receipt_hash=hashlib.sha256((target/'originals/receipt.json').read_bytes()).hexdigest();tables={key:[] for key in ('original_allocations','original_comparisons','budgets','groups','decompositions','diagnostics','reasons')}
    identity=np.broadcast_to(np.arange(5,dtype=np.uint8),(4999,5));biased=np.zeros((4999,5),np.uint8);warn=biased.copy();warn[-1]=4
    for dist,seed in reference.fixed_cells():
        cell=cells[(dist,seed)];panel=reference._allocate(cell,10,20,120);labels=['']
        for oi,sample in enumerate(cell['samples']):
            context=reference.prepare_sample(sample,alpha=.05);means=np.mean(sample[identity],axis=1);records=reference.infer_single(context,means)
            for mi,method in enumerate(audit.METHODS):
                record=records[method]
                for key in (*reference.DIAGNOSTICS,*reference.COUNT_DIAGNOSTICS,'valid','tail_warning'):panel[key][oi,:,mi,:]=record[key]
                if record['reason'] not in labels:labels.append(record['reason'])
                panel['reason_codes'][oi,:,mi,:]=labels.index(record['reason'])
            if dist=='gamma' and seed==0 and oi==0:
                for stream,indices in ((29,biased),(59,warn)):
                    altered=reference.infer_single(context,np.mean(sample[indices],axis=1))
                    for mi,method in enumerate(audit.METHODS):
                        record=altered[method]
                        for key in (*reference.DIAGNOSTICS,*reference.COUNT_DIAGNOSTICS,'valid','tail_warning'):panel[key][oi,0,mi,stream]=record[key]
                        if record['reason'] not in labels:labels.append(record['reason'])
                        panel['reason_codes'][oi,0,mi,stream]=labels.index(record['reason'])
            raw={key:panel[key][oi] for key in (*reference.DIAGNOSTICS,'valid','tail_warning')};result=reference.original_metrics(raw,float(context['se']));local=reference.derived_tables(result,raw,float(context['se']),dist,seed,oi,audit.sample_hash(sample),10,20,120)
            for key,rows in local.items():tables[key].extend(rows)
            panel['observed_streams'][oi]=True
            diag,reasons=reference.diagnostic_rows(panel,oi,labels,dist,seed,20,120,4999);tables['diagnostics'].extend(diag);tables['reasons'].extend(reasons)
            if oi==0:
                for stream in (0,29,30,59,60,119):
                    indices=biased if dist=='gamma' and seed==0 and stream==29 else warn if dist=='gamma' and seed==0 and stream==59 else identity
                    np.savez_compressed(target/'streams'/f'{dist}_seed{seed}_outer0_repeat0_stream{stream}.npz',indices=indices,means=np.mean(sample[indices],axis=1),distribution=dist,master_seed=seed,outer_id=0,budget_repeat_id=0,stream_id=stream,n=5,B=4999,R=20,streams_per_budget=120,alpha=.05,namespace=audit.MAIN_NAMESPACE,native_indices_dtype='int64',stored_indices_dtype='uint8',sample_sha256=cell['sample_sha256'][0])
        panel['reason_labels']=np.array(labels,dtype='U96');reference.save_panel(target/'panels'/f'{dist}_seed{seed}.npz',panel,dist,seed,20,120,4999,original_receipt_sha256=receipt_hash)
    for filename,key in [('original_allocation_metrics.csv','original_allocations'),('original_comparison_metrics.csv','original_comparisons'),('diagnostics_by_original.csv','diagnostics'),('diagnostic_reasons.csv','reasons')]:reference.write_csv(target/filename,tables[key])
    for filename,key in [('budget_metrics.csv.gz','budgets'),('group_metrics.csv.gz','groups'),('decomposition_metrics.csv.gz','decompositions')]:
        with reference.gzip_table(target/filename) as writer:writer.write_rows(tables[key])
    for filename,rows in [('summary.csv',reference.summarize(tables['original_allocations'])),('pooled_summary.csv',reference.summarize(tables['original_allocations'],True)),('decomposition_summary.csv',reference.decomposition_summary(tables['original_comparisons'],tables['decompositions'])),('diagnostic_summary.csv',reference.summarize_diagnostics(tables['diagnostics']))]:reference.write_csv(target/filename,rows)
    reference._write_json(target/'hypotheses.json',reference.assemble_hypotheses(tables['original_comparisons']))
    reference._write_json(target/'resources.json',dict(status='completed',completed_cells=6,incomplete_cells=[],source_git_sha=source_commit,original_receipt_readback_verified=True,original_receipt_sha256=receipt_hash,originals_persisted=60,mother_samples_generated=60,reused_originals=0,observed_inner_streams=144000,observed_logical_intervals=432000,observed_invalid_logical_intervals=sum(row['invalid_streams'] for row in tables['diagnostics']),saved_audit_streams=36))
    before={str(path.relative_to(target)):hashlib.sha256(path.read_bytes()).hexdigest() for path in target.rglob('*') if path.is_file()};result=audit.audit_evidence(target)
    assert result['mismatches']==[]
    assert result['status']=='passed' and result['original_samples']==60 and result['fully_reconstructed_streams']==36
    for name,count in [('original_allocation_metrics.csv',540),('original_comparison_metrics.csv',180),('budget_metrics.csv.gz',10800),('group_metrics.csv.gz',25200),('decomposition_metrics.csv.gz',3600),('summary.csv',54),('pooled_summary.csv',18),('decomposition_summary.csv',24),('diagnostics_by_original.csv',180),('diagnostic_summary.csv',24)]:assert result['table_counts'][name]==dict(reconstructed=count,saved=count)
    assert result['table_counts']['diagnostic_reasons.csv']['reconstructed']==result['table_counts']['diagnostic_reasons.csv']['saved']
    summary=reference.summarize_diagnostics(tables['diagnostics']);bca=next(row for row in summary if row['distribution']=='gamma' and row['master_seed']=='pooled' and row['method']=='bca')
    assert bca['invalid_originals']==2 and bca['warning_streams']>=1 and bca['q_high_one_streams']>=1 and bca['tie_positive_streams']>0
    assert before=={str(path.relative_to(target)):hashlib.sha256(path.read_bytes()).hexdigest() for path in target.rglob('*') if path.is_file()}
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('lo,hi,status',[(-.0024,.0024,'supported'),(-.0025,.0024,'inconclusive'),(-.0024,.0025,'inconclusive'),(.0026,.003,'not_supported'),(-.003,-.0026,'not_supported')])
def test_H1_same_strict_band_exact_contact_no_rounding(lo,hi,status):
    assert audit.mean_status(dict(low=lo,high=hi,defined=True))==status


@pytest.mark.parametrize('llo,lhi,ulo,uhi,defined,q,status',[(.1,.2,-.2,-.1,True,1.,'supported'),(-.1,0,-.2,-.1,True,1.,'not_supported'),(.1,.2,0,.1,True,1.,'not_supported'),(0,.2,-.2,-.1,True,1.,'inconclusive'),(.1,.2,-.2,0,True,1.,'inconclusive'),(-.1,0,-.2,-.1,False,1.,'inconclusive'),(.1,.2,-.2,-.1,True,0.,'inconclusive')])
def test_H2_both_valid_comparisons_positive_averageQ_zero_boundaries(llo,lhi,ulo,uhi,defined,q,status):
    assert audit.variance_status(dict(low=llo,high=lhi,defined=defined),dict(low=ulo,high=uhi,defined=True),q)==status


def comparison_rows():
    rows=[]
    for dist in audit.DISTRIBUTIONS:
        for seed in (0,1,2):
            for outer in range(10):
                q30=.001+outer*.00002+seed*.00001;q120=q30*(1+.02*((outer%3)-1))
                if seed==outer==0:q30=q120=0.
                rows.append(dict(cohort='M10',distribution=dist,master_seed=seed,outer_id=outer,method='bca',complete=True,d=(outer-4.5)*.00003+(seed-1)*.00001,Q30=q30,Q120=q120,L=q120-.8*q30,U=q120-1.25*q30))
    return rows


def test_six_nonzero_CIs_N30_df29_missing_held_and_individual_ratioNA_not_gate():
    from scripts import run_small_sample_benchmark as reference
    rows=comparison_rows();result=audit.hypothesis_reference(rows);assert audit.compare_json(result,reference.assemble_hypotheses(rows))==[]
    assert len(result['quantities'])==6 and all(row['N']==30 and row['df']==29 and row['valid'] for row in result['quantities'].values())
    assert all(row['status']=='supported' for row in result['hypotheses'].values())
    missing=[row for row in rows if not(row['distribution']=='gamma' and row['master_seed']==1 and row['outer_id']==3)]
    for row in missing:
        if row['distribution']=='lognormal':row['Q120']=.5*row['Q30'];row['L']=row['Q120']-.8*row['Q30'];row['U']=row['Q120']-1.25*row['Q30']
    held=audit.hypothesis_reference(missing);assert audit.compare_json(held,reference.assemble_hypotheses(missing))==[]
    assert held['quantities']['H2_L_gamma']['N']==30 and held['quantities']['H2_L_gamma']['finite_count']==29 and held['quantities']['H2_L_gamma']['low'] is None
    assert held['hypotheses']['H2']['status']=='not_supported' and held['population_decisions']['gamma']['H2']['status']=='inconclusive'


def test_logical_diagnostic_counts_missing_invalid_warning_and_t_are_separate():
    from scripts import run_small_sample_benchmark as reference
    panel=fixed_panel(o=1);panel['observed_streams'][0,0,0]=False
    panel['valid'][0,0,2,1]=False;panel['lower'][0,0,2,1]=panel['upper'][0,0,2,1]=np.nan;panel['reason_codes'][0,0,2,1]=1
    panel['tail_warning'][0,0,2,2]=True;panel['q_high'][0,0,2,2]=1;panel['tau_high'][0,0,2,2]=0
    rows,reasons=audit.diagnostic_rows(panel);actual_rows,actual_reasons=reference.diagnostic_rows(panel,0,list(panel['reason_labels']),'gamma',0,20,120,4999)
    assert audit.compare_rows(rows,actual_rows,['distribution','master_seed','outer_id','method'])==[]
    assert audit.compare_rows(reasons,actual_reasons,['distribution','master_seed','outer_id','method','status','reason'])==[]
    summary=audit.summarize_diagnostics(rows);primary=reference.summarize_diagnostics(actual_rows);assert audit.compare_rows(summary,primary,['distribution','master_seed','method'])==[]
    bca=next(row for row in summary if row['distribution']=='gamma' and row['master_seed']==0 and row['method']=='bca')
    assert bca['original_total']==10 and bca['observed_originals']==1 and bca['absent_originals']==9 and bca['missing_originals']==10
    assert bca['invalid_originals']==1 and bca['complete_originals']==0 and bca['incomplete_originals']==10 and bca['warning_streams']==1
    t=next(row for row in rows if row['method']=='t');assert t['planned_streams']==2400 and t['missing_streams']==1
    # These logical diagnostic counts do not replace t's effective independent interval count1.
    assert audit.original_metrics(fixed_panel(o=1))[0]['total_inner']==1


def test_smoke_projection_is_only_six_fixed_M09_outer0_first5_and_main_collision_stops():
    from scripts import run_small_sample_benchmark as reference
    manifest=reference.make_manifest();errors,projections=audit.check_reference(manifest);assert errors==[] and len(projections)==6 and all(row.shape==(5,) for row in projections)
    with np.load(audit.ROOT/'evidence/rq_m09/originals/gamma_seed0.npz',allow_pickle=False) as source:raw=source['raw'][0,:5].copy()
    cell=original_cell();cell['raw'][0]=raw;cell['samples']=audit.transform_raw(cell['raw'],'gamma');cell['sample_sha256']=np.array([audit.sample_hash(row).encode() for row in cell['samples']],dtype='S64');cell['raw_sha256']=np.array([audit.sample_hash(row).encode() for row in cell['raw']],dtype='S64')
    assert np.array_equal(cell['samples'][0],projections[0])
    assert any('smoke projection' in error for error in audit.inspect_original_cell(cell,'gamma',0,projections))
    assert audit.check_registry(manifest)==[]
    for field,value in [('original_key_length',7),('inner_key_length',6),('sample_hashes_known_before_generation',True),('main_namespace',0x4D3039)]:
        damaged=copy.deepcopy(manifest);damaged['generation_registry'][field]=value;assert audit.check_registry(damaged)


def test_exclusive_CLI_and_preregistered_source_guard_mutation(tmp_path,monkeypatch,capsys):
    result=dict(status='passed',mismatch_count=0,mismatches=[],table_counts={});calls=[]
    monkeypatch.setattr(audit,'audit_evidence',lambda path:(calls.append(path) or result));directory=tmp_path/'evidence';directory.mkdir();sentinel=directory/'raw.txt';sentinel.write_bytes(b'preserved')
    assert audit.main([str(directory)])==0 and json.loads(capsys.readouterr().out)['status']=='passed' and list(directory.iterdir())==[sentinel]
    output=tmp_path/'audit.json';assert audit.main([str(directory),'--output',str(output)])==0
    before=len(calls)
    with pytest.raises(SystemExit):audit.main([str(directory),'--output',str(output)])
    assert len(calls)==before and json.loads(output.read_text())==result and sentinel.read_bytes()==b'preserved'
    guard=tmp_path/'scripts/run_small_sample_guarded.py';guard.parent.mkdir();guard.write_bytes(b'fixed guard')
    manifest=dict(source_files_sha256={'scripts/run_small_sample_guarded.py':hashlib.sha256(guard.read_bytes()).hexdigest()});assert audit.check_source_hashes(manifest,root=tmp_path)==[]
    guard.write_bytes(b'mutated guard');assert audit.check_source_hashes(manifest,root=tmp_path)


def test_linear_empirical_q0_q1_are_valid_warning_boundaries_not_exclusions():
    from mathscape import stability
    values=np.linspace(-2,3,4999);record=audit.quantile_record(values,0,1);primary=stability._quantile_record(values,0,1)
    for field in ('lower','upper','valid','reason','q_low','q_high','tau_low','tau_high','tail_warning'):assert record[field]==primary[field]
    assert record['valid'] and record['lower']==-2 and record['upper']==3 and record['tail_warning'] and record['tau_low']==record['tau_high']==0
