"""M09 preflight: deterministic raw/fake-RNG only; NEVER preview main originals."""

from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from mathscape import allocation,matched_moments,replication as r
from scripts import run_allocation_benchmark as old_runner


@pytest.fixture(autouse=True)
def forbid_any_real_random_constructor(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("real M09 random generation forbidden in tests")
    for name in ("SeedSequence","PCG64","Generator","default_rng"):
        monkeypatch.setattr(np.random,name,forbidden)
    monkeypatch.setattr(matched_moments,"generate_sample",forbidden)


class FakeRng:
    """Fixed primitive observations only, unrelated to any M09 random output."""
    def __init__(self,seed=0,outer=0,calls=None):self.seed,self.outer,self.calls=seed,outer,calls
    def gamma(self,shape,scale,size):
        assert (shape,scale,size)==(1.,1.,20)
        if self.calls is not None:self.calls.append(("gamma",self.seed,self.outer))
        return np.linspace(.1,1.9,20)+.03*(10*self.seed+self.outer)
    def standard_normal(self,size):
        assert size==20
        if self.calls is not None:self.calls.append(("lognormal",self.seed,self.outer))
        return np.linspace(-2.2,2.1,20)+.01*(10*self.seed+self.outer)


def reference_fixture():
    # Synthetic OLD observations solely for deterministic collision fixtures.
    cells={}
    for di,dist in enumerate(r.DISTRIBUTIONS):
        for seed in (0,1,2):
            samples=np.array([np.linspace(-100-di*10,-99-di*10,20)-seed-i*.01 for i in range(10)])
            cells[(dist,seed)]=dict(samples=samples,outer_id=np.arange(10,dtype=np.uint16),
                sample_sha256=np.array([r.sample_sha256(x).encode() for x in samples],dtype="S64"))
    return dict(cohort_role="accepted_M08_reference_only",cells=cells,
        reference=dict(source_files_sha256={"fixture":"old_known_hash"}),
        hypotheses=dict(question_id="RQ-M08",hypotheses={"H1":{"status":"supported"}}))


def fake_cohort(monkeypatch,reference=None,progress=None):
    calls=[]
    monkeypatch.setattr(r,"rng_for_original",lambda dist,n,seed,outer,namespace:FakeRng(seed,outer,calls))
    ref=reference or reference_fixture()
    cells=r.generate_cohort(ref,progress=progress)
    assert calls==[(d,s,o) for d in r.DISTRIBUTIONS for s in (0,1,2) for o in range(10)]
    return cells,ref


def test_pure_six_and_seven_keys_registry_has_no_future_hashes_or_attempt_fields():
    assert r.original_key("gamma",20,2,9)==(0x4D3039,1,20,9,0,0)
    assert r.stream_key("lognormal",20,2,9,19,119)==(0x4D3039,2,20,9,1,19,119)
    design=r.generation_design()
    assert len(design["originals"])==60 and not design["sample_hashes_known_before_generation"]
    assert design["original_key_length"]==6 and design["inner_key_length"]==7
    assert len(set(design["namespace_registry"].values()))==len(design["namespace_registry"])
    original_keys={(item['entropy'],tuple(item['spawn_key'])) for item in design['originals']}
    stream_keys={(seed,r.stream_key(dist,20,seed,outer,repeat,stream))
        for dist in r.DISTRIBUTIONS for seed in (0,1,2) for outer in range(10)
        for repeat in range(20) for stream in range(120)}
    assert len(original_keys)==60 and len(stream_keys)==144000 and not original_keys&stream_keys
    assert all(len(key)==6 for _,key in original_keys)
    assert all(len(key)==7 for _,key in stream_keys)
    for item in design['originals']:assert 'sample_sha256' not in item and 'raw_sha256' not in item
    assert 'attempt' not in inspect.signature(r.original_key).parameters


def test_rng_factories_use_exact_entropy_and_key_lengths_with_fake_constructors(monkeypatch):
    calls=[]
    def sequence(*,entropy,spawn_key):
        calls.append((entropy,spawn_key));return (entropy,spawn_key)
    monkeypatch.setattr(np.random,"SeedSequence",sequence)
    monkeypatch.setattr(np.random,"PCG64",lambda sequence:sequence)
    monkeypatch.setattr(np.random,"Generator",lambda bit_generator:dict(fake_bit_generator=bit_generator))
    original=r.rng_for_original("gamma",20,1,4)
    inner=r.rng_for_panel("gamma",20,1,4,1,3,29)
    assert calls==[(1,(r.MAIN_NAMESPACE,1,20,4,0,0)),(1,(r.MAIN_NAMESPACE,1,20,4,1,3,29))]
    assert original['fake_bit_generator']!=inner['fake_bit_generator']


@pytest.mark.parametrize("kwargs",[dict(distribution="normal"),dict(n=19),dict(n=True),
    dict(master_seed=3),dict(master_seed=np.nan),dict(outer_id=10),dict(namespace=allocation.MAIN_NAMESPACE)])
def test_invalid_original_key_rejected_without_rng(kwargs):
    with pytest.raises(ValueError):r.original_key(**({"distribution":"gamma"}|kwargs))


def test_inner_keys_reject_original_purpose_and_out_of_budget_ids_without_rng():
    for kwargs in (dict(purpose_id=0),dict(budget_repeat_id=20),dict(stream_id=120)):
        with pytest.raises(ValueError):r.rng_for_panel("gamma",**kwargs)


@pytest.mark.parametrize("distribution",r.DISTRIBUTIONS)
def test_fixed_raw_transform_exact_accepted_operation_order_no_sample_normalization(distribution):
    record=r.draw_original(distribution,20,1,7,rng=FakeRng(1,7))
    raw=record['raw']
    if distribution=='gamma':
        expected=raw-1.0
        assert record['raw_kind']=='gamma_y' and record['constants']==dict(shape=1.,scale=1.)
    else:
        expected=(np.exp(matched_moments.LOGNORMAL_S*raw)-matched_moments.LOGNORMAL_M)/matched_moments.LOGNORMAL_D
        assert record['raw_kind']=='normal_z'
        assert list(record['constants'])==['v','t','s','m','d']
        for key,value in matched_moments.LOGNORMAL_CONSTANTS.items():
            assert np.float64(record['constants'][key]).tobytes()==np.float64(value).tobytes()
    assert record['sample'].tobytes()==expected.astype('<f8').tobytes()
    assert record['raw'].dtype==record['sample'].dtype==np.dtype('<f8')
    assert record['sample_sha256']==sha256(expected.astype('<f8').tobytes()).hexdigest()
    assert record['raw_sha256']==r.sample_sha256(raw)
    assert not np.isclose(record['sample'].mean(),0)
    assert not np.isclose(record['sample'].var(ddof=1),1)
    assert record['spawn_key']==list(r.original_key(distribution,20,1,7))


def test_smoke_cannot_draw_new_originals_and_constant_sample_is_not_redrawn():
    with pytest.raises(ValueError,match="smoke must reuse"):
        r.draw_original('gamma',namespace=r.SMOKE_NAMESPACE,rng=FakeRng())
    class Constant:
        def gamma(self,**kwargs):return np.ones(20)
    record=r.draw_original('gamma',rng=Constant())
    assert np.all(record['sample']==0)
    assert not r.prepare_sample(record['sample'])['valid']
    assert r.prepare_sample is allocation.prepare_sample and r.infer_single is allocation.infer_single
    assert set(inspect.signature(r.infer_single).parameters)=={'context','means'}


@pytest.mark.parametrize("raw,distribution",[(np.ones(19),'gamma'),(np.full(20,np.nan),'gamma'),
    (-np.ones(20),'gamma'),(np.full(20,1e5),'lognormal')])
def test_raw_invalid_or_transform_nonfinite_stops_no_fallback(raw,distribution):
    with pytest.raises(ValueError):r.transform_raw(raw,distribution)


def test_fake_cohort_all60_once_numeric_schema_and_postgeneration_receipt(monkeypatch):
    progress={};cells,ref=fake_cohort(monkeypatch,progress=progress)
    assert progress['generated_originals']==len(progress['completed_records'])==60
    assert len(cells)==6
    for (dist,seed),cell in cells.items():
        assert cell['samples'].shape==cell['raw'].shape==(10,20)
        assert cell['original_keys'].shape==(10,6) and cell['original_keys'].dtype==np.uint32
        assert cell['sample_sha256'].dtype==cell['raw_sha256'].dtype==np.dtype('S64')
        assert cell['outer_id'].dtype==np.uint16
        assert all(value.dtype.kind!='O' for value in cell.values())
        assert all(not value.flags.writeable for value in cell.values())
    receipt=r.cohort_receipt(cells,ref,'a'*40)
    assert len(receipt['records'])==receipt['original_samples']==60
    assert 'after generation' in receipt['hash_recording_phase']
    assert receipt['isolation_checks']['exact_M08_matches']==receipt['isolation_checks']['duplicate_new_samples']==0
    assert 'not a proof' in receipt['isolation_checks']['interpretation']
    assert all(len(record['spawn_key'])==6 for record in receipt['records'])
    assert '/Users/' not in json.dumps(receipt)


def test_fake_saved_raw_readback_and_validation_never_rng(monkeypatch,tmp_path):
    cells,ref=fake_cohort(monkeypatch)
    folder=tmp_path/'originals';folder.mkdir()
    for (dist,seed),cell in cells.items():np.savez_compressed(folder/f'{dist}_seed{seed}.npz',**cell)
    monkeypatch.setattr(r,'rng_for_original',lambda *a,**kw:pytest.fail('draw during saved read'))
    monkeypatch.setattr(r,'rng_for_panel',lambda *a,**kw:pytest.fail('inner during saved read'))
    loaded=r.load_cohort(folder,ref)
    for key in cells:
        for field in cells[key]:np.testing.assert_array_equal(loaded[key][field],cells[key][field])
    assert r.validate_cohort(loaded,ref)['original_samples']==60
    assert r.cohort_receipt(loaded,ref,'b'*40)['source_git_sha']=='b'*40


@pytest.mark.parametrize('drift',['sample','raw','hash','key_purpose','key_length','constant','IDs','namespace','missing_cell'])
def test_persisted_cohort_drift_is_rejected_no_regeneration(monkeypatch,drift):
    cells,ref=fake_cohort(monkeypatch)
    changed={key:{field:value.copy() for field,value in cell.items()} for key,cell in cells.items()}
    cell=changed[('gamma',0)]
    if drift=='sample':cell['samples'][0,0]+=1
    elif drift=='raw':cell['raw'][0,0]+=1
    elif drift=='hash':cell['sample_sha256'][0]=b'0'*64
    elif drift=='key_purpose':cell['original_keys'][0,4]=1
    elif drift=='key_length':cell['original_keys']=np.pad(cell['original_keys'],((0,0),(0,1)))
    elif drift=='constant':cell['constant_values'][0]=np.nextafter(1.,2.)
    elif drift=='IDs':cell['outer_id'][[0,1]]=cell['outer_id'][[1,0]]
    elif drift=='namespace':cell['namespace']=np.array(allocation.MAIN_NAMESPACE)
    else:changed.pop(('lognormal',2))
    monkeypatch.setattr(r,'rng_for_original',lambda *a,**kw:pytest.fail('repair draw'))
    with pytest.raises(ValueError):r.validate_cohort(changed,ref)


def test_duplicate_or_exact_old_match_stops_without_exchange_preserves_offending_raw(monkeypatch):
    reference=reference_fixture();progress={};calls=[]
    def duplicated(dist,n,master_seed,outer_id,namespace):
        calls.append(outer_id)
        return r.draw_original(dist,n,master_seed,outer_id,namespace,rng=FakeRng(0,0))
    with pytest.raises(ValueError,match='duplicate new-cohort'):
        r.generate_cohort(reference,draw_fn=duplicated,progress=progress)
    assert calls==[0,1] and progress['generated_originals']==2
    assert len(progress['completed_records'])==2 and len(progress['cells'][('gamma',0)]['samples'])==2
    collision=r.draw_original('gamma',rng=FakeRng(0,0))
    reference['cells'][('gamma',0)]['samples'][0]=collision['sample']
    calls.clear();progress={}
    with pytest.raises(ValueError,match='exact M08 sample match'):
        r.generate_cohort(reference,draw_fn=duplicated,progress=progress)
    assert calls==[0] and progress['generated_originals']==1


def test_inner_resampling_uses_fake_indices_int64_and_no_real_rng(monkeypatch):
    calls=[]
    class FakeIndices:
        def integers(self,low,high,size,dtype):
            assert (low,high,dtype)==(0,20,np.int64)
            calls.append(size);return np.arange(np.prod(size),dtype=dtype).reshape(size)%20
    monkeypatch.setattr(r,'rng_for_panel',lambda *args,**kw:FakeIndices())
    sample=np.linspace(-.8,1.2,20)
    means,indices=r.generate_bootstrap_means(sample,'gamma',Bmax=7,max_elements=60,return_indices=True)
    assert calls==[(3,20),(3,20),(1,20)] and indices.dtype==np.uint8
    np.testing.assert_array_equal(means,sample[indices].mean(axis=1))


def comparison_fixture():
    output=[]
    for dist in r.DISTRIBUTIONS:
        for seed in (0,1,2):
            for outer in range(10):
                q=.02+.001*outer
                output.append(dict(distribution=dist,master_seed=seed,outer_id=outer,method='bca',complete=True,
                    d=.0001*(outer-4.5),L=.2*q,U=-.25*q,Q30=q,Q120=q,ratio_Q120_Q30=1.,
                    mean_Jgap=.001,mean_Pgap=-.001,mean_delta=.0001*(outer-4.5)))
    return output


def test_M08_numeric_and_H6_reuse_changes_only_question_ID_old_outcomes_not_input():
    rows=comparison_fixture();old=old_runner.assemble_hypotheses(rows);new=r.assemble_hypotheses(rows)
    assert new==old|dict(question_id='RQ-M09')
    assert new['confirmatory_intervals']==6 and new['decision_confidence']==.995
    assert all(q['N']==30 and q['df']==29 for q in new['quantities'].values())
    reference=reference_fixture();reference['hypotheses']['hypotheses']['H1']['status']='not_supported'
    side=r.cohort_comparison(new,reference)
    assert [c['question_id'] for c in side['cohorts']]==['RQ-M08','RQ-M09']
    assert side['cohorts'][1]['hypotheses']==new and 'no pooling' in side['comparison_role']
    assert r.original_metrics is allocation.original_metrics and r.budget_metrics is allocation.budget_metrics
    # Duplicate IDs would be a pool and are refused, not combined to N60.
    with pytest.raises(ValueError,match='Duplicate'):
        r.assemble_hypotheses(rows+deepcopy(rows))
    missing=[row for row in rows if not(row['distribution']=='gamma' and row['master_seed']==0 and row['outer_id']==0)]
    h=r.assemble_hypotheses(missing)
    assert h['quantities']['H1_gamma']['N']==30 and h['quantities']['H1_gamma']['finite_count']==29
    assert h['population_decisions']['gamma']['H1']['status']=='inconclusive'
    assert not r.mean_t_interval(np.zeros(30))['valid']
    assert r.variance_band_status(-.02,0,-.02,-.01,1)['status']=='not_supported'
    assert r.variance_band_status(-.02,0,0,0,1)['status']=='inconclusive'


def test_accepted_M08_reference_nine_files_hashes_and_smoke_samples_read_only():
    manifest=r.make_reference_manifest();reference=r.load_reference(manifest)
    assert manifest['accepted_head']==r.ACCEPTED_HEAD and len(manifest['source_files_sha256'])==9
    assert not manifest['new_original_hashes_preregistered']
    assert reference['cohort_role']=='accepted_M08_reference_only'
    assert len(reference['pooled'])==18 and len(reference['cells'])==6
    for cell in reference['cells'].values():
        assert cell['samples'].shape==(10,20) and not cell['samples'].flags.writeable
        assert r.sample_sha256(cell['samples'][0]).encode()==cell['sample_sha256'][0]
    assert reference['hypotheses']['question_id']=='RQ-M08'
    with pytest.raises(ValueError,match='M08 reference cells are not main originals'):
        r.validate_cohort(reference['cells'],reference)


def test_M08_reference_Git_byte_drift_stops_without_touching_or_drawing(monkeypatch):
    actual=Path.read_bytes
    def changed(path):
        payload=actual(path)
        return payload+b' ' if path==r.ROOT/'evidence/rq_m08/manifest.json' else payload
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(ValueError,match='differs from accepted Git blob'):r.make_reference_manifest()
