"""M08/M09 saved cohort switching; no source model, inference or RNG execution."""
import base64
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest

from mathscape import (allocation_ui,replication,allocation,bootstrap,bootstrap_ui,distribution_shape,finite_k,inference,inference_ui,matched_moments,matched_stability,stability)
from scripts import run_replication_benchmark

APP=Path(__file__).resolve().parents[1]/'app.py'
MODE='同じ予算で、どう測る？'


@pytest.fixture(autouse=True)
def no_generation_inference_experiments_or_transformation(monkeypatch,caplog):
    def forbidden(*args,**kwargs):raise AssertionError('cohort view may only read saved evidence')
    names=('rng_for_original','rng_for_panel','rng_for_cell','draw_original','generate_cohort','generate_sample','generate_bootstrap_means','bootstrap_means','generate_samples','draw_replicate','intervals',
           'prepare_sample','infer_single','infer_prefixes','original_metrics','replicate_metrics','budget_metrics','t_inference','transform_raw','make_reference_manifest','load_reference','load_cohort','cohort_receipt','validate_cohort','run_cell','run_benchmark','run_smoke')
    for module in (replication,allocation,finite_k,matched_stability,matched_moments,distribution_shape,stability,bootstrap,bootstrap_ui,inference,inference_ui,run_replication_benchmark):
        for name in names:
            if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden);monkeypatch.setattr(policies,'_shown_default_value_warning',False)
    caplog.set_level(logging.WARNING,logger=policies.__name__)
    yield
    assert not any('was created with a default value' in record.getMessage() and 'Session State API' in record.getMessage() for record in caplog.get_records('call'))


def widget(elements,label):return next(item for item in elements if item.label==label)
def healthy(app):assert not app.exception and not app.error


def open_view(cohort='M09'):
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();widget(app.radio,'推論モード').set_value(MODE).run()
    if cohort!='M08':widget(app.selectbox,'保存コホート').set_value(cohort).run()
    healthy(app);return app


def plots(app):return [json.loads(item.proto.spec) for item in app.get('plotly_chart')]
def array(value):
    if isinstance(value,dict) and 'bdata' in value:return np.frombuffer(base64.b64decode(value['bdata']),dtype=value['dtype'])
    return np.asarray(value)
def original(app):return array(plots(app)[0]['data'][0]['x'])


def evidence(cohort):return allocation_ui.EVIDENCE if cohort=='M08' else allocation_ui.NEW_EVIDENCE

def panel(cohort='M09',dist='gamma',seed=0):
    with np.load(evidence(cohort)/'panels'/f'{dist}_seed{seed}.npz',allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}


def test_default_M08_eight_modes_and_new_entry_preserve_existing_contract():
    app=open_view('M08');assert widget(app.selectbox,'保存コホート').value=='M08'
    assert len(widget(app.radio,'推論モード').options)==8 and len(widget(app.radio,'Labを選択').options)==4
    assert any('標本を変えても、同じ結論？' in item.value for item in app.header)
    np.testing.assert_array_equal(original(app),panel('M08')['samples'][0])
    assert allocation_ui.EVIDENCE.name=='rq_m08' and allocation_ui.NEW_EVIDENCE.name=='rq_m09'
    assert widget(app.metric,'旧パネル M08 · H1／H2').value=='支持 ／ 支持' and widget(app.metric,'新標本 M09 · H1／H2').value=='支持 ／ 支持'


def test_M09_defaults_new_raw_receipt_namespace_and_different_original_from_same_M08_ID():
    app=open_view();saved=panel();np.testing.assert_array_equal(original(app),saved['samples'][0])
    with np.load(evidence('M09')/'originals/gamma_seed0.npz',allow_pickle=False) as cell:
        np.testing.assert_array_equal(original(app),cell['samples'][0]);raw=cell['raw'][0].copy()
    assert not np.array_equal(original(app),panel('M08')['samples'][0])
    digest=hashlib.sha256(original(app).astype('<f8').tobytes()).hexdigest();assert digest==saved['sample_sha256'][0].decode()
    assert widget(app.selectbox,'同一予算のmaster seed').value==0 and widget(app.number_input,'同一予算の原標本ID').value==0 and widget(app.selectbox,'予算反復 r').value==0
    assert widget(app.selectbox,'同一予算の手法').value=='bca' and widget(app.selectbox,'同一予算の配分').value=='P30'
    assert any(digest in item.value and 'namespace=0x4D3039' in item.value for item in app.caption)
    receipt=json.loads((evidence('M09')/'originals/receipt.json').read_text())
    assert any(receipt['source_git_sha'] in item.value and '事前固定commit' in item.value and 'の後' in item.value for item in app.caption)
    raw_table=next(item.value for item in app.dataframe if list(item.value.columns)==['observation_index','raw','X'])
    np.testing.assert_array_equal(raw_table.raw,raw);np.testing.assert_array_equal(raw_table.X,original(app))


@pytest.mark.parametrize('dist',['gamma','lognormal'])
def test_M09_allocations_methods_r_keep_hash_and_full_same_original_axis(dist):
    app=open_view();widget(app.selectbox,'同一予算の母集団').set_value(dist).run();saved=panel('M09',dist);before=original(app).copy();axis=plots(app)[0]['layout']['xaxis']['range']
    widget(app.selectbox,'予算反復 r').set_value(19).run()
    for method in ('bca','percentile','t'):
        widget(app.selectbox,'同一予算の手法').set_value(method).run()
        for name,(j,k) in allocation_ui.ALLOCATIONS.items():
            widget(app.selectbox,'同一予算の配分').set_value(name).run();healthy(app)
            np.testing.assert_array_equal(original(app),before);assert plots(app)[0]['layout']['xaxis']['range']==axis
            if method=='t':assert len(plots(app)[0]['data'])==2
            else:
                mi=list(allocation_ui.METHODS).index(method)
                for group,trace in enumerate(plots(app)[0]['data'][2:]):
                    expected=np.array([[saved['lower'][0,19,mi,stream],saved['upper'][0,19,mi,stream],None] for stream in range(group*k,(group+1)*k)]).ravel()
                    assert array(trace['x']).tolist()==expected.tolist()
    for chart in plots(app)[4:6]:
        for trace in chart['data']:np.testing.assert_array_equal(array(trace['y']),np.zeros(3));assert 'error_y' not in trace


def test_cohort_population_ID_seed_r_restore_independently_through_Lab_and_modes():
    app=open_view('M08');widget(app.selectbox,'同一予算のmaster seed').set_value(2).run();widget(app.number_input,'同一予算の原標本ID').set_value(9).run();widget(app.selectbox,'予算反復 r').set_value(7).run();old_gamma=original(app).copy()
    widget(app.selectbox,'保存コホート').set_value('M09').run();assert widget(app.selectbox,'同一予算のmaster seed').value==0 and widget(app.number_input,'同一予算の原標本ID').value==0 and widget(app.selectbox,'予算反復 r').value==0
    widget(app.number_input,'同一予算の原標本ID').set_value(6).run();widget(app.selectbox,'予算反復 r').set_value(13).run();new_gamma=original(app).copy()
    widget(app.selectbox,'同一予算の母集団').set_value('lognormal').run();widget(app.selectbox,'同一予算のmaster seed').set_value(1).run();widget(app.number_input,'同一予算の原標本ID').set_value(8).run();widget(app.selectbox,'予算反復 r').set_value(19).run();new_lognormal=original(app).copy()
    widget(app.selectbox,'同一予算の配分').set_value('P60').run();widget(app.selectbox,'同一予算の手法').set_value('percentile').run()
    widget(app.selectbox,'保存コホート').set_value('M08').run();assert widget(app.number_input,'同一予算の原標本ID').value==0
    widget(app.selectbox,'同一予算の母集団').set_value('gamma').run();np.testing.assert_array_equal(original(app),old_gamma);assert widget(app.number_input,'同一予算の原標本ID').value==9 and widget(app.selectbox,'予算反復 r').value==7
    widget(app.selectbox,'保存コホート').set_value('M09').run();np.testing.assert_array_equal(original(app),new_gamma);assert widget(app.number_input,'同一予算の原標本ID').value==6 and widget(app.selectbox,'予算反復 r').value==13
    widget(app.selectbox,'同一予算の母集団').set_value('lognormal').run();np.testing.assert_array_equal(original(app),new_lognormal)
    widget(app.radio,'推論モード').set_value('揺れの測り方にも揺れがある').run();widget(app.radio,'推論モード').set_value(MODE).run();widget(app.radio,'Labを選択').set_value('Probability Explorer').run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();healthy(app)
    assert widget(app.selectbox,'保存コホート').value=='M09' and widget(app.selectbox,'同一予算のmaster seed').value==1 and widget(app.number_input,'同一予算の原標本ID').value==8 and widget(app.selectbox,'予算反復 r').value==19
    assert widget(app.selectbox,'同一予算の配分').value=='P60' and widget(app.selectbox,'同一予算の手法').value=='percentile';np.testing.assert_array_equal(original(app),new_lognormal)


def test_two_sessions_do_not_mutate_EVIDENCE_or_share_current_cohort():
    original_path=allocation_ui.EVIDENCE;old=open_view('M08');new=open_view('M09');new_hash=original(new).copy();old_hash=original(old).copy()
    widget(old.selectbox,'同一予算の配分').set_value('P60').run();healthy(old)
    widget(new.selectbox,'同一予算の配分').set_value('P120').run();healthy(new)
    assert allocation_ui.EVIDENCE==original_path and allocation_ui.EVIDENCE.name=='rq_m08'
    np.testing.assert_array_equal(original(old),old_hash);np.testing.assert_array_equal(original(new),new_hash);assert not np.array_equal(old_hash,new_hash)
    assert widget(old.selectbox,'保存コホート').value=='M08' and widget(new.selectbox,'保存コホート').value=='M09'


def test_M09_exact_metrics_R20T_and_all18_54_24_saved_tables():
    app=open_view();rows=pd.read_csv(evidence('M09')/'budget_metrics.csv.gz');selected=rows[(rows.distribution=='gamma')&(rows.master_seed==0)&(rows.outer_id==0)&(rows.method=='bca')];current=selected[(selected.budget_repeat_id==0)&(selected.allocation=='P30')].iloc[0]
    for label,key in [('組平均の下端点SD','lower_sd'),('組平均の上端点SD','upper_sd'),('組平均のS','S'),('標本SE（s/√20）','se'),('予算推定値 T=Ase','T'),('平均V=Vbar=Vse','Vbar')]:assert widget(app.metric,label).value==format(current[key],'.6g')
    for trace,name in zip(plots(app)[1]['data'],allocation_ui.ALLOCATIONS):np.testing.assert_allclose(array(trace['y']),selected.loc[selected.allocation==name,'T'],rtol=1e-13)
    for index,name,count in [(5,'pooled_summary.csv',18),(6,'summary.csv',54),(7,'decomposition_summary.csv',24)]:
        actual=app.dataframe[index].value;expected=pd.read_csv(evidence('M09')/name);assert len(actual)==count
        pd.testing.assert_frame_equal(actual.astype(object).where(actual.notna(),None),expected.astype(object).where(expected.notna(),None),check_dtype=False,rtol=1e-12,atol=1e-14)
    assert set(app.dataframe[5].value.cohort)=={'M09'} and set(app.dataframe[5].value.original_total)=={30}
    assert any('新生成した60原標本の1コホート' in item.value for item in app.info)


def test_all_twelve_old_new_intervals_and_bands_exact_saved_not_pooled():
    app=open_view();saved={cohort:json.loads((evidence(cohort)/'hypotheses.json').read_text()) for cohort in ('M08','M09')}
    assert len(plots(app))==11
    for trace,(cohort,dist) in zip(plots(app)[9]['data'],[(c,d) for c in ('M08','M09') for d in allocation_ui.LABELS]):
        row=saved[cohort]['quantities']['H1_'+dist];assert trace['x']==[row['mean']] and trace['error_x']['array']==[row['high']-row['mean']] and trace['error_x']['arrayminus']==[row['mean']-row['low']]
    for trace,(cohort,dist,name) in zip(plots(app)[10]['data'],[(c,d,h) for c in ('M08','M09') for d in allocation_ui.LABELS for h in ('H2_L','H2_U')]):
        row=saved[cohort]['quantities'][name+'_'+dist];assert trace['x']==[row['mean']] and trace['error_x']['array']==[row['high']-row['mean']] and trace['error_x']['arrayminus']==[row['mean']-row['low']]
    band=next(shape for shape in plots(app)[9]['layout']['shapes'] if shape['type']=='rect');assert band['x0']==-.0025 and band['x1']==.0025
    table=next(item.value for item in app.dataframe if {'cohort','quantity','low995','high995'}.issubset(item.value.columns))
    assert len(table)==12 and set(table.N)=={30} and set(table.df)=={29}
    assert any('旧新のpool' in item.value and '新たなコホート間検定' in item.value for item in app.caption)
    assert any('1コホートの限定的な直接追試' in item.value and '独立を証明' in item.value for item in app.caption)
    ratio=plots(app)[8];assert all('error_x' not in trace for trace in ratio['data'])
    assert next(shape for shape in ratio['layout']['shapes'] if shape['type']=='rect')['x0']==.8


def test_selected_cohort_five_payload_bytes_correct_raw_original_namespace_and7200_diags(monkeypatch):
    captured={};real=allocation_ui.st.download_button
    def capture(label,data,**kwargs):captured[kwargs['file_name']]=data;return real(label,data,**kwargs)
    monkeypatch.setattr(allocation_ui.st,'download_button',capture);app=open_view();new_panel=panel()
    for cohort in ('M08','M09'):
        prefix='rq_'+cohort.lower()
        for filename,path in [(prefix+'_pooled_summary.csv','pooled_summary.csv'),(prefix+'_hypotheses.json','hypotheses.json'),(prefix+'_overview.html','figures/overview.html')]:assert captured[filename]==(evidence(cohort)/path).read_bytes()
    assert len(captured)==10 and all(isinstance(value,bytes) and value for value in captured.values())
    rows=list(csv.DictReader(io.StringIO(captured['rq_m09_gamma_seed0_outer0_sample.csv'].decode())));assert len(rows)==20
    values=np.array([float(row['X']) for row in rows],dtype='<f8');np.testing.assert_array_equal(values,original(app));assert hashlib.sha256(values.tobytes()).hexdigest()==rows[0]['sample_sha256']
    with np.load(evidence('M09')/'originals/gamma_seed0.npz',allow_pickle=False) as cell:np.testing.assert_array_equal(np.array([float(row['raw']) for row in rows]),cell['raw'][0]);assert rows[0]['raw_sha256']==cell['raw_sha256'][0].decode()
    assert {row['cohort'] for row in rows}=={'M09'} and {int(row['namespace']) for row in rows}=={0x4D3039}
    endpoints=list(csv.DictReader(io.StringIO(captured['rq_m09_gamma_seed0_outer0_endpoints.csv'].decode())));assert len(endpoints)==7200
    for row in endpoints:
        rep,mi,stream=int(row['budget_repeat_id']),list(allocation_ui.METHODS).index(row['method']),int(row['stream_id']);assert row['cohort']=='M09' and int(row['namespace'])==0x4D3039
        for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high'):
            value,expected=float(row[key]),new_panel[key][0,rep,mi,stream];assert value==expected or (np.isnan(value) and np.isnan(expected))
        assert (row['valid']=='True')==new_panel['valid'][0,rep,mi,stream] and (row['tail_warning']=='True')==new_panel['tail_warning'][0,rep,mi,stream]
    captured.clear();widget(app.selectbox,'保存コホート').set_value('M08').run();assert len(captured)==5 and all(name.startswith('rq_m08_') for name in captured)


def test_legacy_state_migrates_M08_ID_and_free_alpha_is_separate():
    app=AppTest.from_file(str(APP),default_timeout=30)
    app.session_state[allocation_ui.STATE_KEY]=dict(distribution='lognormal',allocation='P60',method='percentile',selections=dict(gamma=dict(seed=0,outer=1,r=2),lognormal=dict(seed=2,outer=7,r=14)))
    app.run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();widget(app.selectbox,'有意水準 α').set_value(.10).run();widget(app.radio,'推論モード').set_value(MODE).run();healthy(app)
    assert widget(app.selectbox,'保存コホート').value=='M08' and widget(app.selectbox,'同一予算のmaster seed').value==2 and widget(app.number_input,'同一予算の原標本ID').value==7 and widget(app.selectbox,'予算反復 r').value==14
    widget(app.selectbox,'保存コホート').set_value('M09').run();assert widget(app.number_input,'同一予算の原標本ID').value==0 and widget(app.selectbox,'予算反復 r').value==0
    assert widget(app.selectbox,'同一予算の配分').value=='P60' and widget(app.selectbox,'同一予算の手法').value=='percentile'
    assert '固定α=.05・B4999' in app.success[0].value
    widget(app.radio,'推論モード').set_value('t推論の探索').run();assert widget(app.selectbox,'有意水準 α').value==.10


@pytest.mark.parametrize('damage',['missing','raw','receipt'])
def test_new_original_corruption_stops_M09_but_does_not_reinterpret_old_cohort(monkeypatch,tmp_path,damage):
    if damage=='missing':monkeypatch.setattr(allocation_ui,'NEW_EVIDENCE',tmp_path)
    elif damage=='raw':
        original_loader=allocation_ui._panel
        def altered(path,modified):
            result=original_loader(path,modified)
            if '/rq_m09/originals/' in path:result['raw']=result['raw'].copy();result['raw'][0,0]+=.01
            return result
        monkeypatch.setattr(allocation_ui,'_panel',altered)
    else:
        original_loader=allocation_ui._panel
        def altered(path,modified):
            result=original_loader(path,modified)
            if '/rq_m09/panels/' in path:result['original_receipt_sha256']=np.array('0'*64)
            return result
        monkeypatch.setattr(allocation_ui,'_panel',altered)
    app=open_view('M08');widget(app.selectbox,'保存コホート').set_value('M09').run()
    assert not app.exception and len(app.error)==1 and not app.get('plotly_chart')
    widget(app.selectbox,'保存コホート').set_value('M08').run();healthy(app);np.testing.assert_array_equal(original(app),panel('M08')['samples'][0])
