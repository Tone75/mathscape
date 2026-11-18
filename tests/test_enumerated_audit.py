"""Deterministic independent-audit fixtures; never analyze M11 real inputs."""
import ast
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
from scripts import audit_enumerated_reference as audit


@pytest.fixture(autouse=True)
def no_random_generation(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('audit/test must never generate randomness')
    for name in ('default_rng','Generator','PCG64','SeedSequence'):monkeypatch.setattr(np.random,name,forbidden)


def test_auditor_has_no_primary_or_other_audit_imports_and_fixed_tolerances():
    tree=ast.parse(Path(audit.__file__).read_text())
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    imports += [item.name for node in ast.walk(tree) if isinstance(node,ast.Import) for item in node.names]
    assert not any('mathscape' in name or 'run_' in name or 'audit_' in name for name in imports if name)
    assert (audit.NUMERIC_RTOL,audit.NUMERIC_ATOL,audit.IDENTITY_RTOL,audit.IDENTITY_ATOL)==(2e-10,2e-12,2e-12,2e-12)


def test_all126_compositions_multinomial3125_positions_not_observation_values():
    result=audit.enumerate_by_composition(np.array([0.,0.,0.,1.,2.]))
    assert result['count_vectors'].shape==(126,5) and result['indices'].shape==(3125,5)
    assert int(result['multinomial_weights'].sum())==3125 and np.all(result['count_vectors'].sum(axis=1)==5)
    assert len(set(map(tuple,result['indices'])))==3125
    for counts,weight in zip(result['count_vectors'],result['multinomial_weights']):assert int(weight)==math.factorial(5)//math.prod(math.factorial(int(v)) for v in counts)
    expected=np.array(list(itertools.product(range(5),repeat=5)),dtype=np.uint8)
    np.testing.assert_array_equal(result['ordered_means'],np.mean(np.array([0.,0.,0.,1.,2.])[expected],axis=1))
    assert result['frequency'].sum()==3125


def test_each_permutation_mean_evaluated_floating_order_effect_not_representative_weighting():
    sample=np.array([1e16,-1e16,1.,2.,4.]);result=audit.enumerate_by_composition(sample)
    rows=np.array(list(audit.position_permutations([1,1,1,1,1])))
    assert np.unique(np.mean(sample[rows],axis=1)).size>1
    direct=np.array(list(itertools.product(range(5),repeat=5)),dtype=np.uint8)
    np.testing.assert_array_equal(result['ordered_means'],np.mean(sample[direct],axis=1))
    support,frequency=np.unique(np.mean(sample[direct],axis=1),return_counts=True)
    np.testing.assert_array_equal(result['support'],support);np.testing.assert_array_equal(result['frequency'],frequency)


@pytest.mark.parametrize('sample',[np.array([-2.,-1.,0.,1.,2.]),np.zeros(5),np.full(5,.8),np.array([-0.,0.,-0.,0.,-0.])])
def test_symmetric_constant_signed_zero_composition_multiset(sample):
    result=audit.enumerate_by_composition(sample);assert result['cumulative'][-1]==3125
    assert result['support'].size==np.unique(result['ordered_means']).size
    assert np.isfinite(result['ordered_means']).all()
    if np.all(sample==sample[0]):assert result['support'].size==1


def test_inverse_cdf_rational_binary_boundary_neighbors_and_endpoints():
    support=np.array([-2.,3.,9.]);frequency=np.array([625,1250,1250])
    for q in (0.,1.,.2,.6,np.nextafter(.2,0),np.nextafter(.2,1),np.nextafter(.6,0),np.nextafter(.6,1)):
        result=audit.inverse_cdf(support,frequency,q);p=Fraction.from_float(q)
        index=next((i for i,c in enumerate(np.cumsum(frequency)) if int(c)*p.denominator>=p.numerator*3125),2)
        assert result['support_index']==index and result['value']==support[index]
        assert Fraction(result['q_numerator'])/Fraction(result['q_denominator'])==p
        assert result['left_count']==(0 if index==0 else int(np.cumsum(frequency)[index-1]))
    # binary64(.2) is strictly greater than exact 1/5, so choose the second atom.
    assert audit.inverse_cdf(support,frequency,.2)['value']==3.
    assert audit.inverse_cdf(support,frequency,np.nextafter(.2,0))['value']==-2.
    assert audit.inverse_cdf(support,frequency,0.)['value']==-2. and audit.inverse_cdf(support,frequency,1.)['value']==9.


@pytest.mark.parametrize('q',[-.01,1.01,np.nan,np.inf])
def test_inverse_cdf_refuses_invalid_q_without_clipping(q):
    with pytest.raises(ValueError):audit.inverse_cdf([0.,1.],np.array([1,1]),q)


def test_adjusted_quantiles_singular_invalid_z_and_exact_endpoint_branch():
    q,reason=audit.adjusted_quantiles(0.,0.);assert reason=='';np.testing.assert_allclose(q,[.025,.975],rtol=0,atol=1e-16)
    assert audit.adjusted_quantiles(np.inf,.01)[1]=='bca_bias_correction_nonfinite'
    assert audit.adjusted_quantiles(0.,np.nan)[1]=='bca_acceleration_undefined'
    assert audit.adjusted_quantiles(0.,1.)[1]=='bca_adjustment_singular'
    assert audit.inverse_cdf([1.,3.],np.array([1,1]),0.)['value']==1.


def test_rmse_bias_ddof0_identity_and_linear_absolute_quantiles():
    delta=np.array([-2.,-1.,.5,4.]);result=audit.error_moments(delta,delta-np.array([.1]*4),np.array([.1]*4))
    assert result['b']==delta.mean() and result['variance_ddof0']==np.var(delta,ddof=0)
    assert result['rmse']==np.sqrt(np.mean(delta**2)) and result['sd_ddof0']==np.sqrt(result['variance_ddof0'])
    assert result['abs_p95']==np.quantile(abs(delta),.95,method='linear')
    assert abs(result['rmse_identity_residual'])<=audit.IDENTITY_ATOL+audit.IDENTITY_RTOL*result['rmse_identity_scale']
    assert result['cross_term']==pytest.approx(2*np.mean((delta-.1)*.1))


def test_planned_denominator_missing_invalid_not_replaced_with_conditional_pass():
    valid=[dict(complete=True,pass_flag=i<27,master_seed=i//10,outer_id=i%10,invalid=False,missing=False) for i in range(30)]
    assert audit.group_decision(valid)['status']=='supported'
    valid[26]['pass_flag']=False;assert audit.group_decision(valid)['status']=='not_supported'
    valid[29]['complete']=False;assert audit.group_decision(valid)['status']=='inconclusive'
    assert audit.group_decision(valid[:29])['status']=='inconclusive'
    assert audit.joint_status(['not_supported','inconclusive'])=='not_supported'
    assert audit.joint_status(['supported','inconclusive'])=='inconclusive'
    assert audit.joint_status([])==audit.joint_status(['supported'])=='inconclusive'


@pytest.mark.parametrize('sample',[np.array([-2.,-1.,0.,1.,2.]),np.array([0.,0.,0.,1.,4.]),np.full(5,.8),np.array([1e16,-1e16,1.,2.,4.]),np.array([-0.,0.,-0.,0.,-0.])])
def test_full_reference_schema_matches_primary_only_on_fixed_fixtures(sample):
    from mathscape import enumerated_reference as primary
    independent=audit.build_reference(sample)
    expected=primary.build_reference(sample)
    audit.assert_arrays(independent,{key:np.asarray(value) for key,value in expected.items()},'fixture_reference',exact_fields=('indices','means','support','frequency','cumulative','E_cdf'))
    if np.all(sample==sample[0]):
        assert not independent['valid'][0] and independent['valid'][1]
        assert independent['reason'][0]=='sample_constant' and np.isnan(independent['se'])


def fixed_saved(reference):
    count=2400
    endpoint=np.broadcast_to(reference['E_cdf'],(count,2,2)).copy()
    # Deterministic deviations on both sides, not a random bootstrap experiment.
    deviations=np.linspace(-.12,.08,count)
    endpoint += deviations[:,None,None]*reference['se']
    return dict(saved_endpoint=endpoint,saved_valid=np.ones((count,2),bool),observed=np.ones(count,bool),
                saved_q=np.broadcast_to(reference['q'],(count,2,2)).copy(),
                saved_z0=np.broadcast_to([reference['z0_star'],np.nan],(count,2)).copy(),
                saved_a=np.full((count,2),reference['a']))


def test_all_streams_metrics_conditional_cross_terms_and_method_local_invalidity():
    from mathscape import enumerated_reference as primary
    reference=audit.build_reference(np.array([-2.,-1.,0.,1.,4.]))
    saved=fixed_saved(reference)
    saved['saved_valid'][0,0]=False
    saved['observed'][1]=False
    independent=audit.stream_comparison(reference,saved)
    expected=primary.stream_comparison(reference,saved)
    audit.assert_arrays(independent,expected,'fixture_comparison')
    assert independent['error_valid'][:,0,0].sum()==2398 and independent['error_valid'][:,1,0].sum()==2399
    for mi in range(2):
        for ei in range(2):
            actual=audit.endpoint_metrics(independent,mi,ei)
            target=primary.endpoint_metrics(expected,mi,ei)
            assert not audit.compare_rows([dict(id=0,**actual)],[dict(id=0,**target)],['id'])
            assert not actual['complete'] and np.isnan(actual['rmse']) and np.isfinite(actual['conditional_rmse'])
            assert actual['missing_series']==1 and actual['invalid_series']==(1 if mi==0 else 0)
    # Removing only BCa finite diagnostics does not poison percentile.
    saved['observed'][:]=True
    saved['saved_valid'][:]=True
    saved['saved_z0'][0,0]=np.inf
    comparison=audit.stream_comparison(reference,saved)
    assert comparison['error_valid'][:,1].all() and not comparison['error_valid'][0,0].any()


@pytest.mark.parametrize('value,passes',[(.1,True),(np.nextafter(.1,0),True),(np.nextafter(.1,1),False)])
def test_threshold_contact_is_exact_no_allclose_or_rounding(value,passes):
    endpoint=dict(complete=True,identity_ok=True,rmse=value,missing_series=0,invalid_series=0)
    result=audit.classify_original([endpoint,endpoint.copy()])
    assert result['pass_flag'] is passes and result['margin']==.1-value


def test_rank_ties_min_max_rule_and_percentile_has_no_scientific_decision():
    rows=[]
    for dist in audit.DISTRIBUTIONS:
        for method in audit.REFERENCE_METHODS:
            for i in range(30):
                rows.append(dict(distribution=dist,method=method,master_seed=i//10,outer_id=i%10,sample_sha256=str(i),e=.05 if i<2 else .2,
                                 margin=.05 if i<2 else -.1,complete=True,pass_flag=i<2,missing=False,invalid=False,rank_e=None))
    examples=audit.rank_and_represent(rows)
    assert len(examples)==4
    for example in examples:
        assert (example['master_seed'],example['outer_id'])==((0,0) if example['selection']=='minimum' else (0,2))
    summaries=audit.summary_rows(rows)
    assert [r['status'] for r in summaries if r['method']=='percentile']==['descriptive_only']*2
    assert set(audit.hypothesis_reference(summaries)['groups'])==set(audit.DISTRIBUTIONS)


def test_wrong_atom_support_or_discrete_order_cannot_hide_inside_float_tolerance():
    expected={'support':np.array([1.,np.nextafter(1.,2.)]),'valid':np.array([True,False]),'atom':np.array([0,1],dtype=np.int64)}
    for field in expected:
        changed={k:v.copy() for k,v in expected.items()}
        changed[field]=changed[field][::-1]
        with pytest.raises(ValueError):audit.assert_arrays(expected,changed,'tampered',exact_fields=('support',))


def test_strict_report_JSON_and_probability_type_validation():
    value=audit.json_safe(dict(missing=np.nan,invalid=np.inf,count=np.int64(3),flag=np.bool_(True)))
    assert json.loads(json.dumps(value,allow_nan=False))==dict(missing=None,invalid=None,count=3,flag=True)
    for q in (True,np.bool_(False),np.array([.5])):
        with pytest.raises(ValueError):audit.inverse_cdf([0.,1.],np.array([1,1]),q)


@pytest.mark.parametrize('quantile',[[-.001,.9],[.1,1.001],[.5,.5],[.8,.2]])
def test_invalid_finite_q_branch_is_local_and_cannot_create_error_measurements(quantile):
    from mathscape import enumerated_reference as primary
    reference=audit.build_reference(np.array([-2.,-1.,0.,1.,4.]))
    saved=fixed_saved(reference)
    saved['saved_q'][0,0]=quantile
    result=audit.stream_comparison(reference,saved)
    audit.assert_arrays(result,primary.stream_comparison(reference,saved),'q_branch')
    assert not result['error_valid'][0,0].any() and result['error_valid'][:,1].all()


def test_finite_streams_with_overflowed_metrics_hold_and_mark_invalid_original():
    from mathscape import enumerated_reference as primary
    comparison=dict(observed=np.ones(2400,bool),error_valid=np.ones((2400,2,2),bool),
                    delta=np.full((2400,2,2),1e200),linear_delta=np.full((2400,2,2),1e200),
                    convention_offset=np.zeros((2400,2,2)))
    measured=audit.endpoint_metrics(comparison,0,0)
    expected=primary.endpoint_metrics(comparison,0,0)
    assert not audit.compare_rows([dict(id=0,**measured)],[dict(id=0,**expected)],['id'])
    assert not measured['complete'] and measured['valid_series']==2400 and measured['invalid_series']==0 and measured['reason']=='derived_metric_nonfinite'
    classification=audit.classify_original([measured,measured.copy()])
    assert classification['invalid'] and not classification['missing'] and classification['pass_flag'] is None
    assert not audit.compare_json(classification,primary.classify_original([measured,measured.copy()]))
    assert classification['reason']=='required_reference_scale_or_series_invalid_or_missing'
