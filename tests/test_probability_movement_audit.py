"""Independent M12 deterministic fixtures; never use actual accepted samples."""
import ast
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import audit_probability_movement as independent


@pytest.fixture(autouse=True)
def no_rng(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('M12 audit fixtures forbid RNG')
    for name in ('SeedSequence','PCG64','Generator','default_rng'):
        monkeypatch.setattr(np.random,name,forbidden)


def test_no_primary_imports_shared_primitive_only_disclosed_and_fixed_tolerance():
    tree=ast.parse(Path(independent.__file__).read_text())
    imports=[node for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any('probability_movement' in (node.module or '') for node in imports)
    shared=[node for node in imports if node.module=='scripts.audit_enumerated_reference']
    assert len(shared)==1 and [name.name for name in shared[0].names]==['reference_intervals']
    assert (independent.NUMERIC_RTOL,independent.NUMERIC_ATOL,independent.IDENTITY_RTOL,independent.IDENTITY_ATOL)==(2e-10,2e-12,2e-12,2e-12)


def ranked_means():return np.repeat(np.array([-2.,3.,9.]),[625,1250,1250])


@pytest.mark.parametrize('q,rank',[(0.,1),(1.,3125),(.2,626),(np.nextafter(.2,0),625),(np.nextafter(.2,1),626),(.6,1875),(np.nextafter(.6,1),1876)])
def test_rank_integer_boundary_neighbors_and_exact_endpoints(q,rank):
    values=ranked_means();selected,observed=independent.select_rank(values,q)
    assert observed==rank and selected==values[rank-1]
    num,den=float(q).as_integer_ratio()
    assert (observed==1 or (observed-1)*den<3125*num) and observed*den>=3125*num


@pytest.mark.parametrize('q',[-.1,1.1,np.nan,np.inf,True,np.array([.2])])
def test_invalid_probability_not_clipped(q):
    with pytest.raises(ValueError):independent.select_rank(ranked_means(),q)


def test_same_atom_different_probability_and_nearby_distinct_atom_are_separate():
    first,_=independent.select_rank(ranked_means(),.3)
    second,_=independent.select_rank(ranked_means(),.4)
    assert first==second==3.
    near=np.nextafter(1.,2.);values=np.r_[np.ones(1500),np.full(1625,near)]
    lower,_=independent.select_rank(values,.4);upper,_=independent.select_rank(values,.6)
    assert np.isclose(lower,upper) and lower!=upper and upper>lower


def test_distribution_saved_multiset_frequency_atom_and_order_are_exact():
    means=ranked_means()[::-1].copy()
    q=np.array([[.2,.6],[.025,.975]])
    ends=np.array([[3.,3.],[-2.,9.]])
    reference=dict(means=means,support=np.array([-2.,3.,9.]),frequency=np.array([625,1250,1250],dtype=np.int64),
                   cumulative=np.array([625,1875,3125],dtype=np.int64),valid=np.ones(2,bool),q=q,E_cdf=ends,atom_index=np.array([[1,1],[0,2]],dtype=np.int64))
    np.testing.assert_array_equal(independent.verify_distribution(reference),ranked_means())
    reference['support'][1]=np.nextafter(3.,4.)
    with pytest.raises(ValueError,match='support'):independent.verify_distribution(reference)


def test_planned_group_denominator_never_shrinks_incomplete_or_missing():
    rows=[dict(master_seed=i//10,outer_id=i%10,complete=True,pass_flag=i<27,invalid=False,missing=False) for i in range(30)]
    assert independent.population_decision(rows)['status']=='supported'
    rows[26]['pass_flag']=False;assert independent.population_decision(rows)['status']=='not_supported'
    rows[-1]['complete']=False;result=independent.population_decision(rows)
    assert result['status']=='inconclusive' and result['planned_N']==30 and result['unknown_N']==1
    assert independent.population_decision(rows[:-1])['planned_N']==30
    assert independent.joint_status(['not_supported','inconclusive'])=='not_supported'


def test_strict_json_nonfinite_NA_and_exact_flag_numeric_comparison():
    payload=independent.json_safe(dict(value=np.nan,inf=np.inf,count=np.int64(4),flag=np.bool_(False)))
    assert json.loads(json.dumps(payload,allow_nan=False))==dict(value=None,inf=None,count=4,flag=False)
    independent.assert_value(np.nan,'NA','missing')
    with pytest.raises(ValueError):independent.assert_value(False,True,'flag')
    with pytest.raises(ValueError):independent.assert_value(0,1,'count')
    with pytest.raises(ValueError):independent.assert_arrays({'atom':np.array([1.])},{'atom':np.array([np.nextafter(1.,2.)])},'exact',exact_fields=('atom',))


def test_existing_archived_indices_bridge_checks_means_probabilities_and_intervals(tmp_path):
    sample=np.array([-1.,-.5,0.,.7,2.])
    indices=np.tile(np.arange(5,dtype=np.uint8),(4999,1))
    means=np.mean(sample[indices],axis=1)
    records=independent.reference_intervals(sample,means)
    panel=dict(samples=sample[None,:],reason_labels=np.array(['']))
    for field in records['t']:
        if field=='reason':continue
        dtype=bool if field in ('valid','tail_warning') else np.uint16 if field in ('less_count','equal_count') else float
        panel[field]=np.array([records[method][field] for method in ('t','percentile','bca')],dtype=dtype).reshape(1,1,3,1)
    panel['reason_codes']=np.zeros((1,1,3,1),dtype=np.uint8)
    archive=dict(indices=indices,means=means,distribution='gamma',master_seed=0,outer_id=0,budget_repeat_id=0,stream_id=0,
                 n=5,B=4999,R=20,streams_per_budget=120,alpha=.05,namespace=0x4D3130,
                 native_indices_dtype='int64',stored_indices_dtype='uint8',sample_sha256=independent.sample_hash(sample))
    path=tmp_path/'invented_stream.npz';np.savez_compressed(path,**archive)
    independent.verify_archived_stream(path,panel,'gamma',0,0)
    panel['q_high'][0,0,2,0]+=.001
    with pytest.raises(ValueError,match='q_high'):independent.verify_archived_stream(path,panel,'gamma',0,0)
    archive['means']=means.copy();archive['means'][0]+=.001;np.savez_compressed(path,**archive)
    with pytest.raises(ValueError,match='means differ'):independent.verify_archived_stream(path,panel,'gamma',0,0)


def artificial_pair():
    means=ranked_means()
    reference=dict(means=means,support=np.array([-2.,3.,9.]),frequency=np.array([625,1250,1250],np.int64),
                   cumulative=np.array([625,1875,3125],np.int64),valid=np.ones(2,bool),reason=np.array(['','']),
                   q=np.array([[.2,.8],[.025,.975]]),E_cdf=np.array([[3.,9.],[-2.,9.]]),atom_index=np.array([[1,2],[0,2]],np.int64),se=1.)
    q=np.broadcast_to(reference['q'],(2400,2,2)).copy()
    q[::3,0,0]=.1;q[1::3,0,0]=.3
    Y=np.broadcast_to(reference['E_cdf'],(2400,2,2)).copy()+.01
    comparison=dict(saved_q=q,saved_endpoint=Y,saved_valid=np.ones((2400,2),bool),observed=np.ones(2400,bool),
                    delta=Y-reference['E_cdf'],error_valid=np.ones((2400,2,2),bool),saved_reason=np.full((2400,2),'',dtype='U96'),
                    saved_warning=np.zeros((2400,2),bool),saved_tau_low=np.full((2400,2),100.),saved_tau_high=np.full((2400,2),100.))
    comparison['saved_warning'][0,0]=True
    return reference,comparison


def test_all_array_schema_and_endpoint_stats_independent_rank_matches_primary_fixed_fixture():
    from mathscape import probability_movement as primary
    reference,comparison=artificial_pair()
    data=independent.map_original(reference,comparison)
    expected=primary.analyze_sample(reference,comparison)
    independent.assert_arrays(data,expected,'fixed_mapping',exact_fields=('C','A','percentile_C','percentile_A'))
    assert data['movement_code'][0,0]==-1 and data['movement_code'][1,0]==0
    assert data['q_changed_same'][1,0] and np.all(data['percentile_g']==0)
    for ei in range(2):
        measured=independent.endpoint_metrics(data,ei)
        independent.assert_value(measured,primary.endpoint_metrics(expected,ei),'metrics')
    assert np.isnan(independent.endpoint_metrics(data,1)['moved_abs_g_mean'])


@pytest.mark.parametrize('damage',['missing','saved_invalid','unordered_q','q_outside','q_nonfinite','bad_Y','old_delta_invalid','pct_invalid'])
def test_invalid_reasons_method_endpoint_masks_do_not_shrink_planned_or_cross_methods(damage):
    from mathscape import probability_movement as primary
    reference,comparison=artificial_pair()
    if damage=='missing':comparison['observed'][0]=False
    elif damage=='saved_invalid':comparison['saved_valid'][0,0]=False;comparison['saved_reason'][0,0]='fixture_invalid'
    elif damage=='unordered_q':comparison['saved_q'][0,0]=[.8,.2]
    elif damage=='q_outside':comparison['saved_q'][0,0]=[-.1,.9]
    elif damage=='q_nonfinite':comparison['saved_q'][0,0,0]=np.nan
    elif damage=='bad_Y':comparison['saved_endpoint'][0,0]=[9.,-2.]
    elif damage=='old_delta_invalid':comparison['error_valid'][0,0,0]=False
    else:comparison['saved_valid'][0,1]=False;comparison['saved_reason'][0,1]='fixture_pct_invalid'
    data=independent.map_original(reference,comparison)
    independent.assert_arrays(data,primary.analyze_sample(reference,comparison),'invalid_arrays',exact_fields=('C','movement_code','selected_index'))
    if damage=='pct_invalid':assert data['valid'].all()
    else:assert not data['valid'][0,0]
    row=independent.endpoint_metrics(data,0)
    independent.assert_value(row,primary.endpoint_metrics(data,0),'invalid_metrics')
    assert row['planned_series']==2400
    if damage!='pct_invalid':assert np.isnan(row['rms_g']) and np.isfinite(row['conditional_rms_g'])


def test_signed_cross_term_cancellation_scale_and_zero_movement_reference():
    reference,comparison=artificial_pair()
    data=independent.map_original(reference,comparison)
    data['g'][:]=1e10;data['h'][:]=-1e10+1.;data['d'][:]=1.
    data['q_delta'][:]=.01;data['movement_code'][:]=1
    data['total_error_residual'][:]=0.
    row=independent.endpoint_metrics(data,0)
    assert row['cross_term']<0 and row['moment_identity_scale']>=abs(row['cross_term']) and row['identity_ok']
    assert row['moment_identity_residual']==row['mean_d2']-(row['mean_g2']+row['mean_h2']+row['cross_term'])


@pytest.mark.parametrize('u,passes',[(.1,True),(np.nextafter(.1,0),True),(np.nextafter(.1,1),False)])
def test_threshold_contact_is_not_an_audit_tolerance(u,passes):
    row=dict(complete=True,identity_ok=True,rms_g=u,missing_series=0,invalid_series=0)
    observed=independent.classify_original([row,row.copy()])
    assert observed['pass_flag'] is passes and observed['margin']==.1-u


@pytest.mark.parametrize('invalid',['scale_zero','scale_nan','reference_invalid'])
def test_reference_scale_invalid_reason_is_local_and_preserves_unknown_points(invalid):
    from mathscape import probability_movement as primary
    reference,comparison=artificial_pair()
    if invalid=='reference_invalid':
        reference['valid'][0]=False;reference['reason']=np.array(['fixture_invalid_reference',''])
    else:reference['se']=0. if invalid=='scale_zero' else np.nan
    result=independent.map_original(reference,comparison)
    independent.assert_arrays(result,primary.analyze_sample(reference,comparison),'reference_invalid')
    assert not result['valid'].any() and np.all(result['movement_code']==-2)
    assert np.isnan(result['C']).all()
    if invalid=='reference_invalid':assert result['percentile_valid'].all()


def test_reference_atom_and_fixed_control_are_integrity_checks_not_new_hypotheses():
    reference,comparison=artificial_pair()
    reference['E_cdf'][0,0]=np.nextafter(reference['E_cdf'][0,0],np.inf)
    with pytest.raises(ValueError,match='reference'):independent.map_original(reference,comparison)
    reference,comparison=artificial_pair();comparison['saved_q'][0,1,0]=.5
    with pytest.raises(ValueError,match='percentile'):independent.map_original(reference,comparison)


def test_cli_exclusive_output_and_failed_record_without_scientific_run(monkeypatch,tmp_path):
    output=tmp_path/'invented_result';output.mkdir()
    calls=[]
    def passed(root,destination):
        calls.append((root,destination));return dict(status='passed',mismatch_count=0,scope='fake only')
    monkeypatch.setattr(independent,'audit',passed)
    assert independent.main(['--root',str(tmp_path),'--output',str(output)])==0
    report=output/'independent_audit.json';before=report.read_bytes()
    assert json.loads(before)['status']=='passed' and calls==[(tmp_path,output)]
    with pytest.raises(FileExistsError):independent.main(['--root',str(tmp_path),'--output',str(output)])
    assert report.read_bytes()==before and len(calls)==1
    failed=tmp_path/'invented_failure';failed.mkdir()
    def mismatch(*args):raise ValueError('fixture_mismatch')
    monkeypatch.setattr(independent,'audit',mismatch)
    assert independent.main(['--root',str(tmp_path),'--output',str(failed)])==1
    assert json.loads((failed/'independent_audit.json').read_bytes())==dict(status='failed',mismatch_count=1,question_id='RQ-M12',error_type='ValueError',error='fixture_mismatch')
