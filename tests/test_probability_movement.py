"""M12 deterministic invented arrays only; never maps actual accepted q values."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mathscape import probability_movement as m
from scripts import analyze_probability_movement as worker


@pytest.fixture(autouse=True)
def no_rng_or_old_science(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No RNG or real science run in M12 fixtures')
    for name in ('SeedSequence', 'PCG64', 'Generator', 'default_rng'):
        monkeypatch.setattr(np.random, name, forbidden)


def fixture():
    support = np.array([0., 1., 2., 3.])
    frequency = np.array([625, 937, 938, 625], np.int64)
    reference = dict(support=support, frequency=frequency,
        cumulative=np.cumsum(frequency), se=1., q=np.array([[.25,.75],[.025,.975]]),
        E_cdf=np.array([[1.,2.],[0.,3.]]), atom_index=np.array([[1,2],[0,3]],np.int64),
        valid=np.ones(2,bool), reason=np.array(['',''],dtype='U96'))
    q = np.broadcast_to(reference['q'],(2400,2,2)).copy()
    y = np.broadcast_to(reference['E_cdf'],(2400,2,2)).copy()
    y[:,:,0] += .01
    y[:,:,1] -= .02
    comparison = dict(saved_q=q, saved_endpoint=y, saved_valid=np.ones((2400,2),bool),
        observed=np.ones(2400,bool), delta=(y-reference['E_cdf'])/reference['se'],
        error_valid=np.ones((2400,2,2),bool), saved_reason=np.full((2400,2),'',dtype='U96'),
        saved_warning=np.zeros((2400,2),bool),
        saved_tau_low=4999*np.minimum(q[:,:,0],1-q[:,:,0]),
        saved_tau_high=4999*np.minimum(q[:,:,1],1-q[:,:,1]))
    return reference,comparison


def record(distribution='gamma', seed=0, outer=0):
    reference,comparison=fixture()
    return dict(distribution=distribution,master_seed=seed,outer_id=outer,
                sample_sha256='synthetic_'+distribution,reference=reference,comparison=comparison,
                m11_complete=True,m11_pass=True,m11_e=.08)


def test_exact_cumulative_binary_boundary_and_nextafter_neighbors():
    cumulative=np.array([625,3125],np.int64)
    assert m.cumulative_index(cumulative,.2)==1
    assert m.cumulative_index(cumulative,np.nextafter(.2,0))==0
    assert m.cumulative_index(cumulative,np.nextafter(.2,1))==1
    assert m.cumulative_index(cumulative,0)==0
    assert m.cumulative_index(cumulative,1)==1
    for q in (-.1,1.1,np.nan,np.inf,True):
        with pytest.raises(ValueError):m.cumulative_index(cumulative,q)


def test_distribution_positive_integer_mass_and_exact_support_contract():
    r,_=fixture()
    m.validate_distribution(r['support'],r['frequency'],r['cumulative'])
    for frequencies in (np.array([625,937,938,624],np.int64),np.array([625,937,938,625.]),np.array([0,937,938,1250],np.int64)):
        with pytest.raises(ValueError):m.validate_distribution(r['support'],frequencies,r['cumulative'])
    with pytest.raises(ValueError):m.validate_distribution(np.array([0.,1.,1.,3.]),r['frequency'],r['cumulative'])


def test_q_changed_same_atom_distinct_from_actual_distance_and_zero_moved_NA():
    ref,saved=fixture()
    saved['saved_q'][:,0]=[.26,.74]
    data=m.analyze_sample(ref,saved)
    assert data['q_changed'].all() and data['q_changed_same'].all()
    assert np.all(data['movement_code']==0) and np.all(data['g']==0)
    metric=m.endpoint_metrics(data,0)
    assert metric['same_count']==2400 and metric['moved_count']==0
    assert metric['q_changed_same_count']==2400 and metric['same_given_q_changed_rate']==1
    assert metric['rms_g']==0 and np.isnan(metric['moved_abs_g_mean'])
    assert np.all(data['percentile_g']==0)


def test_same_down_up_counts_and_moved_only_mean_are_different_denominators():
    ref,saved=fixture()
    saved['saved_q'][0,0]=[.1,.9]
    saved['saved_q'][1,0]=[.26,.74]
    data=m.analyze_sample(ref,saved)
    assert data['movement_code'][0].tolist()==[-1,1]
    metric=m.endpoint_metrics(data,0)
    assert metric['same_count']==2399 and metric['down_count']==1 and metric['up_count']==0
    assert metric['moved_abs_g_mean']==1
    assert metric['mean_abs_g']==pytest.approx(1/2400)
    assert metric['q_changed_count']==2 and metric['q_changed_same_count']==1
    assert metric['q_changed_same_rate']==pytest.approx(1/2400)
    assert metric['same_given_q_changed_rate']==.5


def test_adjacent_binary64_support_is_moved_even_when_isclose():
    ref,saved=fixture()
    ref['support']=np.array([0.,1.,np.nextafter(1.,2.),2.,3.])
    ref['frequency']=np.full(5,625,np.int64)
    ref['cumulative']=np.cumsum(ref['frequency'])
    ref['E_cdf']=np.array([[1.,2.],[0.,3.]])
    ref['atom_index']=np.array([[1,3],[0,4]],np.int64)
    saved['delta']=(saved['saved_endpoint']-ref['E_cdf'])/ref['se']
    saved['saved_q'][0,0,0]=.45
    data=m.analyze_sample(ref,saved)
    assert np.isclose(data['C'][0,0],data['A'][0])
    assert data['C'][0,0]!=data['A'][0]
    assert data['movement_code'][0,0]==1 and data['support_step'][0,0]==1
    assert not data['q_changed_same'][0,0] and data['g'][0,0]>0


def test_q0_q1_valid_warning_included_without_exclusion():
    ref,saved=fixture()
    saved['saved_q'][0,0]=[0.,1.]
    saved['saved_warning'][0,0]=True
    saved['saved_tau_low'][0,0]=0
    saved['saved_tau_high'][0,0]=0
    data=m.analyze_sample(ref,saved)
    assert data['valid'][0].all() and data['C'][0].tolist()==[0.,3.]
    assert m.endpoint_metrics(data,0)['source_warning_series']==1


@pytest.mark.parametrize('badq',[[np.nan,.9],[-.1,.9],[.9,.8],[.5,.5],[.1,1.1]])
def test_bad_probability_pair_has_reason_local_NA_not_clipping(badq):
    ref,saved=fixture()
    saved['saved_q'][0,0]=badq
    data=m.analyze_sample(ref,saved)
    assert not data['valid'][0].any()
    assert np.all(data['movement_code'][0]==-2)
    assert np.isnan(data['g'][0]).all()
    reason=data['reason_labels'][data['reason_code'][0,0]]
    assert reason=='saved_probability_nonfinite_outside_or_unordered'
    assert data['percentile_valid'].all()
    metric=m.endpoint_metrics(data,0)
    assert not metric['complete'] and metric['valid_series']==2399
    assert np.isnan(metric['rms_g']) and np.isfinite(metric['conditional_rms_g'])


def test_missing_vs_saved_invalid_reasons_and_percentile_locality():
    ref,saved=fixture()
    saved['observed'][0]=False
    saved['saved_valid'][1,0]=False
    saved['saved_reason'][1,0]='bca_bias_correction_nonfinite'
    saved['saved_valid'][2,1]=False
    saved['saved_reason'][2,1]='bootstrap_distribution_nonfinite'
    data=m.analyze_sample(ref,saved)
    metric=m.endpoint_metrics(data,0)
    assert metric['missing_series']==1 and metric['invalid_series']==1
    assert data['valid'][2].all() and not data['percentile_valid'][2].any()
    assert data['reason_labels'][data['reason_code'][0,0]]=='not_observed'
    assert data['reason_labels'][data['reason_code'][1,0]]=='M10_saved_invalid:bca_bias_correction_nonfinite'
    assert data['q_changed'][0].tolist()==[False,False]


def test_reference_invalid_and_SE_invalid_preserve_required_unknowns():
    ref,saved=fixture()
    ref['valid'][0]=False
    ref['reason'][0]='bca_adjustment_singular'
    data=m.analyze_sample(ref,saved)
    assert not data['valid'].any() and data['percentile_valid'].all()
    assert m.classify_original([m.endpoint_metrics(data,0),m.endpoint_metrics(data,1)])['invalid']
    ref,saved=fixture()
    ref['se']=0
    data=m.analyze_sample(ref,saved)
    assert not data['valid'].any() and not data['percentile_valid'].any()
    assert data['reason_labels'][data['reason_code'][0,0]]=='sample_SE_invalid'


def test_reference_endpoint_and_M11_direct_total_mismatch_stop():
    ref,saved=fixture()
    ref['E_cdf'][0,0]+=1e-6
    with pytest.raises(ValueError,match='Q\\(q\\*\\)'):m.analyze_sample(ref,saved)
    ref,saved=fixture()
    saved['delta'][0,0,0]+=.01
    with pytest.raises(ArithmeticError,match='total error'):m.analyze_sample(ref,saved)


def test_percentile_g_nonzero_is_implementation_check_not_new_hypothesis():
    ref,saved=fixture()
    saved['saved_q'][0,1]=[.4,.6]
    with pytest.raises(ArithmeticError,match='percentile'):m.analyze_sample(ref,saved)


def test_cross_term_cancellation_kept_and_identity_scale_explicit():
    ref,saved=fixture()
    data=m.analyze_sample(ref,saved)
    data['g'][:]=1e8
    data['h'][:]=-1e8+.1
    data['d'][:]=.1
    data['q_delta'][:]=.01
    data['abs_g'][:]=1e8
    data['movement_code'][:]=1
    data['total_error_residual'][:]=0
    metric=m.endpoint_metrics(data,0)
    assert metric['complete'] and metric['identity_ok']
    assert metric['cross_term']<0 and metric['mean_g2']>1e15
    assert metric['mean_d2']<.02
    assert metric['moment_identity_scale']>=abs(metric['cross_term'])


def test_derived_overflow_is_unknown_not_false_complete():
    ref,saved=fixture()
    data=m.analyze_sample(ref,saved)
    data['g'][:]=1e200
    data['h'][:]=-1e200
    data['d'][:]=0
    data['movement_code'][:]=1
    metric=m.endpoint_metrics(data,0)
    assert not metric['complete'] and metric['reason']=='derived_metric_nonfinite'
    assert np.isnan(metric['rms_g'])
    original=m.classify_original([metric,metric])
    assert original['invalid'] and original['pass_flag'] is None


def test_exact_threshold_contact_nextafter_and27of30_with_NA_hold():
    endpoint=dict(complete=True,identity_ok=True,rms_g=.1,missing_series=0,invalid_series=0)
    assert m.classify_original([endpoint,endpoint])['pass_flag'] is True
    assert m.classify_original([endpoint,endpoint|dict(rms_g=np.nextafter(.1,np.inf))])['pass_flag'] is False
    rows=[dict(master_seed=i//10,outer_id=i%10,complete=True,pass_flag=i<27,invalid=False,missing=False) for i in range(30)]
    assert m.group_decision(rows)['status']=='supported'
    rows[26]['pass_flag']=False
    assert m.group_decision(rows)['status']=='not_supported'
    rows[-1].update(complete=False,pass_flag=None,invalid=True)
    group=m.group_decision(rows)
    assert group['status']=='inconclusive' and group['planned_N']==30 and group['unknown_N']==1
    assert m.joint_status(['not_supported','inconclusive'])=='not_supported'
    assert m.joint_status(['supported','inconclusive'])=='inconclusive'


def test_classification_four_cells_IDs_and_unknowns_not_forced_false():
    rows=[]
    for i,(old,new) in enumerate(((False,False),(False,True),(True,False),(True,True))):
        rows.append(dict(distribution='gamma',master_seed=0,outer_id=i,sample_sha256='fixed',
            m11_complete=True,m11_pass=old,pass_flag=new,complete=True,reason=''))
    rows.append(rows[-1]|dict(outer_id=4,complete=False,pass_flag=None,reason='missing'))
    cells,ids=worker.classifications(rows)
    assert len(cells)==8 and all(r['count']==1 for r in cells if r['distribution']=='gamma')
    assert ids[-1]['category']=='unknown' and ids[-1]['m12_pass'] is None
    assert cells[0]['ids']=='0:0'


def test_synthetic_writer_compact_arrays_tables_controls_and_no_overwrite(tmp_path):
    result=worker.analyze_records([record()],tmp_path)
    assert result['status']=='inconclusive' and result['groups']['gamma']['planned_N']==30
    with np.load(tmp_path/'series/gamma_seed0_outer0.npz',allow_pickle=False) as data:
        assert data['g'].shape==(2400,2)
        assert data['movement_code'].dtype==np.int8 and data['selected_index'].dtype==np.int32
        assert data['reason_code'].dtype==np.uint8
        np.testing.assert_array_equal(data['cdf_probability'],data['cumulative']/3125)
        assert np.all(data['percentile_g']==0)
    assert len(list(csv.DictReader((tmp_path/'endpoint_metrics.csv').open())))==2
    assert len(list(csv.DictReader((tmp_path/'summary.csv').open())))==4
    resources=json.loads((tmp_path/'analysis_resources.json').read_text())
    assert resources['status']=='completed' and resources['BCa_endpoints']==4800
    with pytest.raises(FileExistsError):worker.analyze_records([record()],tmp_path)


def test_all60_input_barrier_before_mapping(monkeypatch,tmp_path):
    for name in ('manifest.json','input_receipt.json'):(tmp_path/name).write_text('{}')
    monkeypatch.setattr(worker,'validate_inputs',lambda root:(_ for _ in ()).throw(ValueError('final ID differs')))
    monkeypatch.setattr(m,'analyze_sample',lambda *args:pytest.fail('no mapping before all60valid'))
    with pytest.raises(ValueError):worker.analyze(tmp_path,tmp_path)
    assert not (tmp_path/'series').exists()


def test_missing_immutable_receipt_stops_before_any_source_reads(monkeypatch,tmp_path):
    monkeypatch.setattr(worker,'validate_inputs',lambda *args:pytest.fail('receipt first'))
    with pytest.raises(ValueError,match='receipt'):worker.analyze(tmp_path,tmp_path)


def test_final_write_failure_is_stopped_not_completed(monkeypatch,tmp_path):
    original=worker.write_json
    def failure(path,payload):
        if Path(path).name=='decision.json':raise OSError('synthetic output failure')
        return original(path,payload)
    monkeypatch.setattr(worker,'write_json',failure)
    with pytest.raises(OSError):worker.analyze_records([record()],tmp_path)
    resources=json.loads((tmp_path/'analysis_resources.json').read_text())
    assert resources['completed_originals']==1 and resources['status']=='stopped'
