"""M10 preflight uses fixed observations/fake generators, never real PRNG draws."""
from copy import deepcopy
from hashlib import sha256
import inspect
import json

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr,ndtri

from mathscape import allocation,matched_moments,replication,small_sample as s


@pytest.fixture(autouse=True)
def no_actual_random_constructor(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("No actual M10 random draw in tests")
    for name in ("SeedSequence","PCG64","Generator","default_rng"):
        monkeypatch.setattr(np.random,name,forbidden)
    monkeypatch.setattr(matched_moments,"generate_sample",forbidden)


class FixedRaw:
    def __init__(self,seed=0,outer=0,calls=None):self.seed,self.outer,self.calls=seed,outer,calls
    def gamma(self,shape,scale,size):
        assert (shape,scale,size)==(1.,1.,5)
        if self.calls is not None:self.calls.append(("gamma",self.seed,self.outer))
        return np.array([.2,.4,.8,1.4,2.4])+.013*(self.outer+10*self.seed)
    def standard_normal(self,size):
        assert size==5
        if self.calls is not None:self.calls.append(("lognormal",self.seed,self.outer))
        return np.array([-1.4,-.5,.1,.7,1.6])+.007*(self.outer+10*self.seed)


def reference_fixture():
    cells={}
    for di,d in enumerate(s.DISTRIBUTIONS):
        for seed in (0,1,2):
            x=np.array([np.linspace(-200-di*50,-190-di*50,20)-seed-o*.01 for o in range(10)])
            cells[(d,seed)]=dict(samples=x,outer_id=np.arange(10,dtype=np.uint16),
                sample_sha256=np.array([s.sample_sha256(row).encode() for row in x],dtype="S64"))
    return dict(cohort_role="accepted_M09_reference_only",cells=cells,
        reference=dict(source_files_sha256={"fixture":"known"}),hypotheses=dict(question_id="RQ-M09"))


def make_fake_cohort(reference=None,progress=None,existing_cells=None,calls=None):
    reference=reference or reference_fixture()
    def fake(d,n,master_seed,outer_id,namespace):
        return s.draw_original(d,n,master_seed,outer_id,namespace,rng=FixedRaw(master_seed,outer_id,calls))
    return s.generate_cohort(reference,draw_fn=fake,progress=progress,existing_cells=existing_cells),reference


def subset(cells,selection):
    result={}
    for identity,ids in selection.items():
        cell=cells[identity]
        result[identity]={k:(v[ids].copy() if k in ("samples","raw","sample_sha256","raw_sha256","outer_id","original_keys") else v.copy()) for k,v in cell.items()}
    return result


def endpoint_fixture():
    x=np.arange(120,dtype=float)
    lo=np.vstack([np.full(120,-.7),-.8+.003*x+.02*np.sin(x),-.9+.004*x+.01*np.cos(x)])
    hi=np.vstack([np.full(120,1.2),1+.005*x+.02*np.cos(x),1.1+.007*x+.02*np.sin(x)])
    return dict(lower=lo,upper=hi,valid=np.ones((3,120),bool),tail_warning=np.zeros((3,120),bool))


def test_exact_key_registry_all144060_identities_no_future_sample_hashes():
    design=s.generation_design()
    assert s.original_key("gamma",5,2,9)==(0x4D3130,1,5,9,0,0)
    assert s.stream_key("lognormal",5,2,9,19,119)==(0x4D3130,2,5,9,1,19,119)
    identities={(r['entropy'],tuple(r['spawn_key'])) for r in design['originals']}
    identities.update((seed,s.stream_key(d,5,seed,o,r,k)) for d in s.DISTRIBUTIONS for seed in (0,1,2)
        for o in range(10) for r in range(20) for k in range(120))
    assert len(identities)==design['all_identity_count']==144060
    assert {len(key) for _,key in identities}=={6,7}
    namespaces=design['namespace_registry']
    assert len(set(namespaces.values()))==18 and namespaces['M09_main']==replication.MAIN_NAMESPACE
    assert namespaces['M10_main']==0x4D3130 and namespaces['M10_smoke']==0x534D3130
    assert not design['sample_hashes_known_before_generation']
    assert all('sample_sha256' not in r and 'raw_sha256' not in r for r in design['originals'])
    assert not {'method','J','K','batch','attempt'}&inspect.signature(s.stream_key).parameters.keys()


def test_fake_factories_verify_entropy_and_exact_six_vs_seven_key(monkeypatch):
    calls=[]
    monkeypatch.setattr(np.random,'SeedSequence',lambda **kw:calls.append(kw) or kw)
    monkeypatch.setattr(np.random,'PCG64',lambda seq:seq)
    monkeypatch.setattr(np.random,'Generator',lambda bg:bg)
    s.rng_for_original('gamma',5,1,4)
    s.rng_for_panel('lognormal',5,2,3,1,8,59)
    assert calls==[dict(entropy=1,spawn_key=(0x4D3130,1,5,4,0,0)),dict(entropy=2,spawn_key=(0x4D3130,2,5,3,1,8,59))]


@pytest.mark.parametrize('kwargs',[dict(n=20),dict(n=True),dict(master_seed=3),dict(outer_id=10),
    dict(namespace=replication.MAIN_NAMESPACE),dict(namespace=replication.SMOKE_NAMESPACE)])
def test_wrong_n_old_namespace_or_bad_identity_refused_without_rng(kwargs):
    with pytest.raises(ValueError):s.original_key('gamma',**kwargs)


def test_inner_original_purpose_and_bad_r_k_refused_before_rng():
    for kw in (dict(purpose_id=0),dict(budget_repeat_id=20),dict(stream_id=120)):
        with pytest.raises(ValueError):s.rng_for_panel('gamma',**kw)


@pytest.mark.parametrize('dist',s.DISTRIBUTIONS)
def test_fixed_raw_exact_accepted_transform_constant_bits_and_hash(dist):
    record=s.draw_original(dist,5,1,7,rng=FixedRaw(1,7))
    raw=record['raw']
    expected=raw-1 if dist=='gamma' else (np.exp(matched_moments.LOGNORMAL_S*raw)-matched_moments.LOGNORMAL_M)/matched_moments.LOGNORMAL_D
    assert record['sample'].tobytes()==expected.astype('<f8').tobytes()
    assert record['sample'].shape==record['raw'].shape==(5,)
    assert record['sample_sha256']==sha256(expected.astype('<f8').tobytes()).hexdigest()
    assert record['raw_sha256']==s.sample_sha256(raw)
    assert record['constants']==matched_moments.population_info(dist)['constants']
    for k,v in record['constants'].items():assert np.float64(v).tobytes()==np.float64(matched_moments.population_info(dist)['constants'][k]).tobytes()
    assert not np.isclose(record['sample'].mean(),0) and not np.isclose(record['sample'].var(ddof=1),1)
    assert record['spawn_key']==list(s.original_key(dist,5,1,7))


def test_n5_t_interval_df4_se_and_jackknife_acceleration_reference():
    sample=np.array([-.8,-.3,.2,.7,2.1]);c=s.prepare_sample(sample)
    se=sample.std(ddof=1)/np.sqrt(5);margin=stats.t.ppf(.975,4)*se
    assert c['se']==pytest.approx(se)
    assert c['t']['lower']==pytest.approx(sample.mean()-margin)
    assert c['t']['upper']==pytest.approx(sample.mean()+margin)
    jack=np.array([np.delete(sample,i).mean() for i in range(5)])
    diff=jack.mean()-jack
    a=np.sum(diff**3)/(6*np.sum(diff**2)**1.5)
    assert c['a']==pytest.approx(a,abs=1e-15)
    assert not {'distribution','true_mean','sigma'}&inspect.signature(s.prepare_sample).parameters.keys()
    assert not {'distribution','true_mean','sigma'}&inspect.signature(s.infer_single).parameters.keys()


def test_b4999_half_tie_bias_linear_quantiles_independent_formula():
    sample=np.array([-2.,-1.,0.,1.,2.]);c=s.prepare_sample(sample)
    means=np.resize(np.array([-.7,-.3,0.,0.,.2,.5,.8]),4999)
    result=s.infer_single(c,means);counts=s.tie_diagnostics(c,means)
    less=np.count_nonzero(means<c['mean']);equal=np.count_nonzero(means==c['mean'])
    p=(less+.5*equal)/4999;z0=ndtri(p);z=stats.norm.ppf([.025,.975])
    q=ndtr(z0+(z0+z)/(1-c['a']*(z0+z)))
    assert result['bca']['valid']
    np.testing.assert_allclose([result['bca']['lower'],result['bca']['upper']],np.quantile(means,q,method='linear'),atol=1e-15)
    assert result['bca']['z0']==z0 and counts['bias_probability']==p
    assert counts['less_count']==less and counts['equal_count']==counts['tie_count']==equal
    assert sum(counts[k] for k in ('less_count','equal_count','greater_count'))==4999
    for method in s.METHODS:
        assert result[method]['less_count']==less and result[method]['equal_count']==equal
        old=allocation.infer_single(c,means)[method]
        for k,v in old.items():
            if isinstance(v,float) and np.isnan(v):assert np.isnan(result[method][k])
            else:assert result[method][k]==v


def test_warning_q_endpoints_are_valid_not_excluded_or_clipped():
    # Legitimate n5 acceleration from a fixed sample. This intentionally fixed
    # possible bootstrap-mean vector reaches q=1, not a random pilot.
    c=s.prepare_sample(np.array([-1.,-1.,-1.,-1.,4.]))
    means=np.r_[np.full(4998,-1.),4.]
    rec=s.infer_single(c,means)['bca']
    assert rec['valid'] and rec['q_high']==1
    assert rec['tail_warning'] and rec['tau_high']==0
    assert rec['upper']==4.  # Preserve the endpoint, never clip/fallback.
    c=s.prepare_sample(np.array([-2.,-1.,0.,1.,2.]))
    out=s.infer_single(c,np.linspace(-.5,.5,99))
    assert out['bca']['valid'] and out['bca']['tail_warning']
    assert out['bca']['tau_low']<10 and out['bca']['tau_high']<10


def test_constant_s0_nonfinite_and_true_bca_invalid_preserve_t_and_counts():
    c=s.prepare_sample(np.full(5,.8));assert not c['valid']
    out=s.infer_single(c,np.zeros(4999))
    assert all(not v['valid'] and v['less_count']==s.UNKNOWN_COUNT for v in out.values())
    c=s.prepare_sample(np.array([-2.,-1.,0.,1.,2.]))
    out=s.infer_single(c,np.full(4999,5.))
    assert out['t']['valid'] and out['percentile']['valid']
    assert not out['bca']['valid'] and out['bca']['reason']=='bca_bias_correction_nonfinite'
    assert out['bca']['less_count']==out['bca']['equal_count']==0
    out=s.infer_single(c,np.array([0.,np.nan]))
    assert out['t']['valid'] and not out['bca']['valid'] and out['bca']['less_count']==s.UNKNOWN_COUNT


def test_apop_n5_only_math_change_allocation_pairing_and_exact_t_zero():
    raw=endpoint_fixture();se=.31
    current=s.budget_metrics(raw,se);accepted=allocation.budget_metrics(raw,se)
    for layer in current:
        for field in current[layer]:
            expected=accepted[layer][field]*.5 if layer=='allocations' and field=='Apop' else accepted[layer][field]
            if np.asarray(expected).dtype.kind in 'f':np.testing.assert_allclose(current[layer][field],expected,rtol=0,atol=0,equal_nan=True)
            else:np.testing.assert_array_equal(current[layer][field],expected)
    np.testing.assert_allclose(current['allocations']['Apop'],current['allocations']['S']*np.sqrt(5),rtol=1e-15)
    raw={k:np.stack([v,v]) for k,v in raw.items()};out=s.original_metrics(raw,se,expected_repeats=2)
    assert np.all(out['original_allocations']['M'][:,0]==0) and np.all(out['original_allocations']['Q'][:,0]==0)
    assert out['original_comparisons']['d'][0]==0 and np.isnan(out['original_comparisons']['ratio_Q120_Q30'][0])
    assert np.all(out['original_allocations']['total_streams'][:,0]==1)
    np.testing.assert_allclose(out['original_allocations']['Apop'],out['original_allocations']['S']*np.sqrt(5))
    raw['valid'][1,1,-1]=False
    missing=s.original_metrics(raw,se,expected_repeats=2)
    assert not missing['original_comparisons']['complete'][1] and missing['original_comparisons']['complete'][2]
    assert np.isnan(missing['original_comparisons']['d'][1])


def test_m10_reuses_fixed_hypothesis_thresholds_missing_and_se0_contract():
    assert s.band_status(-.0024,.0024)['status']=='supported'
    for a,b in ((-.0025,.0024),(-.0024,.0025)):
        assert s.band_status(a,b)['status']=='inconclusive'
    assert s.band_status(.0026,.0028)['status']=='not_supported'
    assert s.band_status(-.0024,.0024,complete=False)['status']=='inconclusive'
    assert not s.mean_t_interval(np.full(30,.8),.995)['valid']
    assert not s.mean_t_interval(np.r_[np.arange(29.),np.nan],.995)['valid']
    assert s.variance_band_status(1,2,-2,-1,.1)['status']=='supported'
    assert s.variance_band_status(-1,0,-2,-1,.1)['status']=='not_supported'
    assert s.variance_band_status(1,2,0,1,.1)['status']=='not_supported'
    assert s.variance_band_status(0,1,-1,0,.1)['status']=='inconclusive'
    assert s.variance_band_status(1,2,-2,-1,0)['status']=='inconclusive'
    assert s.variance_band_status(1,2,-2,-1,.1,complete=False)['status']=='inconclusive'


def test_smoke_projection_exact_parent_and_projection_hashes_no_rng():
    ref=reference_fixture();projection=s.smoke_projection(ref)
    assert len(projection)==6
    for key,cell in projection.items():
        expected=ref['cells'][key]['samples'][0,:5]
        assert cell['samples'].shape==(1,5) and cell['samples'].tobytes()==expected.astype('<f8').tobytes()
        assert cell['sample_sha256'][0].decode()==s.sample_sha256(expected)
        assert cell['source_sample_sha256'][0]==ref['cells'][key]['sample_sha256'][0]
    with pytest.raises(ValueError):s.validate_cohort(projection,ref)
    with pytest.raises(ValueError):s.draw_original('gamma',namespace=s.SMOKE_NAMESPACE,rng=FixedRaw())


def test_full_fake_cohort_receipt_shape_n5_and_pure_saved_readback(tmp_path):
    progress={};calls=[];cells,ref=make_fake_cohort(progress=progress,calls=calls)
    assert len(calls)==60 and len(set(calls))==60
    assert progress['generated_originals']==progress['completed_originals']==60 and progress['reused_originals']==0
    receipt=s.cohort_receipt(cells,ref,'a'*40)
    assert receipt['original_samples']==60 and receipt['status']=='complete'
    assert all(r['n']==5 and len(r['spawn_key'])==6 for r in receipt['records'])
    assert not receipt['isolation_checks']['exact_smoke_projection_matches']
    for (d,seed),cell in cells.items():
        assert cell['samples'].shape==cell['raw'].shape==(10,5)
        assert cell['original_keys'].shape==(10,6)
        assert all(a.dtype.kind!='O' and not a.flags.writeable for a in cell.values())
        np.savez_compressed(tmp_path/f'{d}_seed{seed}.npz',**cell)
    loaded=s.load_cohort(tmp_path,ref)
    assert s.cohort_receipt(loaded,ref,'a'*40)==receipt
    assert '/Users/' not in json.dumps(receipt)


def test_partial_reuse_arbitrary_ids_draws_only_missing_preserves_every_byte(tmp_path):
    full,ref=make_fake_cohort()
    saved=subset(full,{('gamma',0):[0,3,9],('gamma',2):[4],('lognormal',1):[1,8]})
    receipt=s.cohort_receipt(saved,ref,'a'*40,partial=True)
    assert receipt['status']=='partial' and receipt['original_samples']==6
    for (d,seed),cell in saved.items():np.savez_compressed(tmp_path/f'{d}_seed{seed}.npz',**cell)
    restored=s.load_cohort(tmp_path,ref,allow_partial=True)
    progress={};calls=[];resumed,_=make_fake_cohort(ref,progress,restored,calls)
    assert progress['generated_originals']==54 and progress['reused_originals']==6 and progress['completed_originals']==60
    assert len(calls)==54 and not set(calls)&{(d,seed,int(o)) for (d,seed),cell in saved.items() for o in cell['outer_id']}
    for key in full:
        for field in full[key]:assert resumed[key][field].tobytes()==full[key][field].tobytes()
    with pytest.raises(ValueError):s.load_cohort(tmp_path,ref)


@pytest.mark.parametrize('tamper',['sample','raw','hash','key','constant','id','namespace'])
def test_preserved_partial_tamper_refused_before_any_missing_draw(tamper):
    cells,ref=make_fake_cohort();saved=subset(cells,{('gamma',0):[3]});cell=saved[('gamma',0)]
    if tamper=='sample':cell['samples'][0,0]+=.1
    elif tamper=='raw':cell['raw'][0,0]+=.1
    elif tamper=='hash':cell['sample_sha256'][0]=b'0'*64
    elif tamper=='key':cell['original_keys'][0,-1]=1
    elif tamper=='constant':cell['constant_values'][0]=2.
    elif tamper=='id':cell['outer_id'][0]=11
    else:cell['namespace']=np.array(s.SMOKE_NAMESPACE)
    def forbidden(*a,**kw):pytest.fail('must validate all existing cells before any new draw')
    with pytest.raises(ValueError):s.generate_cohort(ref,existing_cells=saved,draw_fn=forbidden)


def test_exact_smoke_or_new_duplicate_stops_and_preserves_offending_raw():
    ref=reference_fixture();progress={};calls=[]
    def duplicate(d,n,master_seed,outer_id,namespace):
        calls.append(outer_id)
        return s.draw_original(d,n,master_seed,outer_id,namespace,rng=FixedRaw())
    with pytest.raises(ValueError,match='duplicate'):s.generate_cohort(ref,draw_fn=duplicate,progress=progress)
    assert len(calls)==2 and progress['generated_originals']==2
    assert progress['cells'][('gamma',0)]['raw'].shape==(2,5)
    record=s.draw_original('gamma',rng=FixedRaw())
    ref['cells'][('gamma',0)]['samples'][0,:5]=record['sample']
    ref['cells'][('gamma',0)]['sample_sha256'][0]=s.sample_sha256(ref['cells'][('gamma',0)]['samples'][0]).encode()
    with pytest.raises(ValueError,match='smoke projection'):s.generate_cohort(ref,draw_fn=duplicate,progress={})


def test_fake_indices_b4999_dtype_batch_and_optional_saved_indices(monkeypatch):
    class Indices:
        def __init__(self):self.calls=[];self.offset=0
        def integers(self,low,high,size,dtype):
            assert (low,high,dtype)==(0,5,np.int64)
            self.calls.append(size);a=np.arange(self.offset,self.offset+np.prod(size),dtype=dtype).reshape(size)%5;self.offset+=np.prod(size);return a
    rng=Indices();monkeypatch.setattr(s,'rng_for_panel',lambda *a,**kw:rng)
    sample=np.array([-.8,-.2,.1,.7,2.])
    means,indices=s.generate_bootstrap_means(sample,'gamma',Bmax=4999,max_elements=1000,return_indices=True)
    assert means.shape==(4999,) and indices.shape==(4999,5) and indices.dtype==np.uint8
    assert max(a*b for a,b in rng.calls)<=1000
    np.testing.assert_array_equal(means,sample[indices].mean(axis=1))


def test_accepted_m09_reference_gitblob_readonly_n20_and_smoke_n5():
    reference=s.load_reference();manifest=s.make_reference_manifest()
    assert manifest==reference['reference'] and manifest['accepted_head']==s.ACCEPTED_HEAD
    assert len(manifest['source_files_sha256'])==9 and not manifest['new_original_hashes_preregistered']
    assert all(c['samples'].shape==(10,20) for c in reference['cells'].values())
    assert all(c['samples'].shape==(1,5) for c in s.smoke_projection(reference).values())
