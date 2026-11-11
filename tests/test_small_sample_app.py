"""Saved n5 cohort: exact evidence, no scientific computation, durable state."""
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
from mathscape import (allocation_ui,small_sample,replication,allocation,bootstrap,finite_k,inference,
                       matched_moments,matched_stability,stability,distribution_shape)
from scripts import run_small_sample_benchmark,run_replication_benchmark,run_allocation_benchmark

APP=Path(__file__).resolve().parents[1]/'app.py'
MODE='同じ予算で、どう測る？'


@pytest.fixture(autouse=True)
def saved_only(monkeypatch,caplog):
    def forbidden(*args,**kwargs):raise AssertionError('saved UI cannot generate, transform, infer or run experiments')
    names=('rng_for_original','rng_for_panel','rng_for_cell','draw_original','generate_cohort','generate_sample','generate_bootstrap_means','bootstrap_means','generate_samples','draw_replicate','intervals',
           'prepare_sample','infer_single','infer_prefixes','original_metrics','replicate_metrics','budget_metrics','t_inference','transform_raw','make_reference_manifest','load_reference','load_cohort','cohort_receipt','validate_cohort','run_cell','run_benchmark','run_smoke','smoke_projection')
    for module in (small_sample,replication,allocation,bootstrap,finite_k,inference,matched_moments,matched_stability,stability,distribution_shape,run_small_sample_benchmark,run_replication_benchmark,run_allocation_benchmark):
        for name in names:
            if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
    for name in ('default_rng','SeedSequence','PCG64','Generator'):monkeypatch.setattr(np.random,name,forbidden)
    monkeypatch.setattr(policies,'_shown_default_value_warning',False);caplog.set_level(logging.WARNING,logger=policies.__name__)
    yield
    assert not any('was created with a default value' in record.getMessage() and 'Session State API' in record.getMessage() for record in caplog.get_records('call'))


def widget(elements,label):return next(item for item in elements if item.label==label)
def healthy(app):assert not app.exception and not app.error

def open_view(cohort='M10'):
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();widget(app.radio,'推論モード').set_value(MODE).run()
    if cohort!='M08':widget(app.selectbox,'保存コホート').set_value(cohort).run()
    healthy(app);return app

def plots(app):return [json.loads(item.proto.spec) for item in app.get('plotly_chart')]
def array(value):
    if isinstance(value,dict) and 'bdata' in value:return np.frombuffer(base64.b64decode(value['bdata']),dtype=value['dtype'])
    return np.asarray(value)
def original(app):return array(plots(app)[0]['data'][0]['x'])
def panel(dist='gamma',seed=0):
    with np.load(allocation_ui.SMALL_EVIDENCE/'panels'/f'{dist}_seed{seed}.npz',allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}

def test_n5_entry_defaults_df4_fixed_alpha_and_saved_original():
    app=open_view();saved=panel();np.testing.assert_array_equal(original(app),saved['samples'][0]);assert len(original(app))==5
    assert len(widget(app.radio,'推論モード').options)==8 and len(widget(app.radio,'Labを選択').options)==4
    assert any('標本が5個でも、同じ結論？' in item.value for item in app.header)
    assert 'n5・固定α=.05・B4999・R20' in app.success[0].value
    assert any('t df4' in item.value and 'df29' in item.value for item in app.caption)
    assert widget(app.metric,'小標本 M10（n5） · H1／H2').value=='保留 ／ 保留'
    assert any('非同等を証明したわけではありません' in item.value for item in app.markdown)
    assert any('pool・ID対応比較・n20対n5の差の検定を行いません' in item.value for item in app.caption)
    assert widget(app.selectbox,'同一予算の手法').value=='bca' and widget(app.selectbox,'同一予算の配分').value=='P30'
    assert set(app.dataframe[-1].value.K)=={1} and app.dataframe[-1].value[['lower_sd','upper_sd','S','Ase','Apop']].isna().all().all()


@pytest.mark.parametrize('dist',['gamma','lognormal'])
def test_all_methods_allocations_r_preserve_five_observations_and_common_axis(dist):
    app=open_view();widget(app.selectbox,'同一予算の母集団').set_value(dist).run();saved=panel(dist);before=original(app).copy();axis=plots(app)[0]['layout']['xaxis']['range']
    widget(app.selectbox,'予算反復 r').set_value(19).run()
    for method in allocation_ui.METHODS:
        widget(app.selectbox,'同一予算の手法').set_value(method).run()
        for name,(j,k) in allocation_ui.ALLOCATIONS.items():
            widget(app.selectbox,'同一予算の配分').set_value(name).run();healthy(app);np.testing.assert_array_equal(original(app),before);assert plots(app)[0]['layout']['xaxis']['range']==axis
            assert len(array(plots(app)[0]['data'][0]['y']))==5
            if method=='t':assert len(plots(app)[0]['data'])==2
            else:
                mi=list(allocation_ui.METHODS).index(method)
                for group,trace in enumerate(plots(app)[0]['data'][2:]):
                    expected=np.array([[saved['lower'][0,19,mi,k0],saved['upper'][0,19,mi,k0],None] for k0 in range(group*k,(group+1)*k)]).ravel()
                    assert array(trace['x']).tolist()==expected.tolist()
    widget(app.selectbox,'同一予算の手法').set_value('t').run()
    for chart in plots(app)[4:6]:
        for trace in chart['data']:np.testing.assert_array_equal(array(trace['y']),np.zeros(3));assert 'error_y' not in trace


def test_six_cohort_population_IDs_restore_global_method_allocation_and_lab_mode():
    app=open_view('M08');expected={}
    for ci,cohort in enumerate(('M08','M09','M10')):
        widget(app.selectbox,'保存コホート').set_value(cohort).run()
        for di,dist in enumerate(allocation_ui.LABELS):
            widget(app.selectbox,'同一予算の母集団').set_value(dist).run();widget(app.selectbox,'同一予算のmaster seed').set_value(di).run();widget(app.number_input,'同一予算の原標本ID').set_value(ci+di+1).run();widget(app.selectbox,'予算反復 r').set_value(ci*3+di).run();expected[cohort,dist]=original(app).copy()
    widget(app.selectbox,'同一予算の配分').set_value('P60').run();widget(app.selectbox,'同一予算の手法').set_value('percentile').run()
    widget(app.radio,'推論モード').set_value('揺れの測り方にも揺れがある').run();widget(app.radio,'推論モード').set_value(MODE).run();widget(app.radio,'Labを選択').set_value('Chaos Explorer').run();widget(app.radio,'Labを選択').set_value('Inference Lab').run()
    for ci,cohort in enumerate(('M08','M09','M10')):
        widget(app.selectbox,'保存コホート').set_value(cohort).run()
        for di,dist in enumerate(allocation_ui.LABELS):
            widget(app.selectbox,'同一予算の母集団').set_value(dist).run();healthy(app);np.testing.assert_array_equal(original(app),expected[cohort,dist]);assert widget(app.number_input,'同一予算の原標本ID').value==ci+di+1 and widget(app.selectbox,'予算反復 r').value==ci*3+di
            assert widget(app.selectbox,'同一予算の配分').value=='P60' and widget(app.selectbox,'同一予算の手法').value=='percentile'


def test_sessions_and_legacy_M09_state_migration_are_isolated():
    old=AppTest.from_file(str(APP),default_timeout=30)
    old.session_state[allocation_ui.STATE_KEY]=dict(cohort='M09',distribution='lognormal',allocation='P60',method='percentile',selections=allocation_ui._defaults(),cohort_selections={'M08':allocation_ui._defaults(),'M09':dict(gamma=dict(seed=0,outer=0,r=0),lognormal=dict(seed=2,outer=7,r=14))})
    old.run();widget(old.radio,'Labを選択').set_value('Inference Lab').run();widget(old.selectbox,'有意水準 α').set_value(.10).run();widget(old.radio,'推論モード').set_value(MODE).run();healthy(old)
    assert widget(old.number_input,'同一予算の原標本ID').value==7
    widget(old.selectbox,'保存コホート').set_value('M10').run();assert widget(old.number_input,'同一予算の原標本ID').value==0
    other=open_view('M08');before=original(old).copy();widget(other.number_input,'同一予算の原標本ID').set_value(9).run();widget(old.selectbox,'同一予算の配分').set_value('P120').run();np.testing.assert_array_equal(original(old),before)
    assert allocation_ui.EVIDENCE.name=='rq_m08' and allocation_ui.NEW_EVIDENCE.name=='rq_m09' and widget(other.selectbox,'保存コホート').value=='M08'
    widget(old.radio,'推論モード').set_value('t推論の探索').run();assert widget(old.selectbox,'有意水準 α').value==.10


def test_exact_six_metrics_and_all_18_54_24_and_diagnostic24_tables():
    app=open_view();base=allocation_ui.SMALL_EVIDENCE;budgets=pd.read_csv(base/'budget_metrics.csv.gz');row=budgets.query("distribution=='gamma' and master_seed==0 and outer_id==0 and method=='bca' and allocation=='P30' and budget_repeat_id==0").iloc[0]
    for label,key in [('組平均の下端点SD','lower_sd'),('組平均の上端点SD','upper_sd'),('組平均のS','S'),('標本SE（s/√5）','se'),('予算推定値 T=Ase','T'),('平均V=Vbar=Vse','Vbar')]:assert widget(app.metric,label).value==format(row[key],'.6g')
    for index,name,count in [(5,'pooled_summary.csv',18),(6,'summary.csv',54),(7,'decomposition_summary.csv',24)]:
        actual=app.dataframe[index].value;expected=pd.read_csv(base/name);assert len(actual)==count
        pd.testing.assert_frame_equal(actual.astype(object).where(actual.notna(),None),expected.astype(object).where(expected.notna(),None),check_dtype=False,rtol=1e-12,atol=1e-14)
    diag=next(item.value for item in app.dataframe if 'tie_known_streams' in item.value.columns)
    expected=pd.read_csv(base/'diagnostic_summary.csv');pd.testing.assert_frame_equal(diag.astype(object).where(diag.notna(),None),expected.astype(object).where(expected.notna(),None),check_dtype=False,rtol=1e-12,atol=1e-14)
    selected=budgets.query("distribution=='gamma' and master_seed==0 and outer_id==0 and method=='bca'")
    for trace,allocation in zip(plots(app)[1]['data'],allocation_ui.ALLOCATIONS):np.testing.assert_allclose(array(trace['y']),selected.loc[selected.allocation==allocation,'T'],rtol=1e-12,atol=1e-14)
    groups=pd.read_csv(base/'group_metrics.csv.gz').query("distribution=='gamma' and master_seed==0 and outer_id==0 and method=='bca' and budget_repeat_id==0")
    np.testing.assert_allclose(app.dataframe[2].value[['lower_var','upper_var','V','A']].astype(float),groups[['lower_var','upper_var','V','A']],rtol=1e-12,atol=1e-14)
    decomposition=pd.read_csv(base/'decomposition_metrics.csv.gz').query("distribution=='gamma' and master_seed==0 and outer_id==0 and method=='bca' and budget_repeat_id==0").iloc[0]
    np.testing.assert_allclose(array(plots(app)[3]['data'][0]['y']),[decomposition.Jgap,decomposition.Pgap],rtol=1e-12,atol=1e-14)
    pooled=diag[(diag.master_seed=='pooled')&(diag.method=='bca')];assert pooled.warning_streams.tolist()==[16,1] and pooled.invalid_streams.sum()==0 and pooled.missing_streams.sum()==0
    assert any('有効警告と無効は別です' in item.value for item in app.caption)
    assert any('tau_low=B×q_low、tau_high=B×(1−q_high)' in item.value for item in app.caption)


def test_all18_comparison_intervals_exact_and_fixed_bands_no_new_comparison():
    app=open_view();saved={c:json.loads((allocation_ui._evidence(c)/'hypotheses.json').read_text()) for c in allocation_ui.COHORTS}
    table=next(item.value for item in app.dataframe if {'cohort','quantity','low995','high995'}.issubset(item.value.columns));assert len(table)==18 and set(table.N)=={30} and set(table.df)=={29}
    for row in table.to_dict('records'):
        fixed=saved[row['cohort']]['quantities'][row['quantity']+'_'+row['distribution']]
        for key,source in [('mean','mean'),('low995','low'),('high995','high'),('MCSE','mcse')]:assert row[key]==fixed[source]
    for chart,names in zip(plots(app)[9:11],(('H1',),('H2_L','H2_U'))):
        for trace,(c,d,h) in zip(chart['data'],[(c,d,h) for c in allocation_ui.COHORTS for d in allocation_ui.LABELS for h in names]):
            fixed=saved[c]['quantities'][h+'_'+d];assert trace['x']==[fixed['mean']] and trace['error_x']['array']==[fixed['high']-fixed['mean']]
    band=next(s for s in plots(app)[9]['layout']['shapes'] if s['type']=='rect');assert (band['x0'],band['x1'])==(-.0025,.0025)
    assert all('error_x' not in trace for trace in plots(app)[8]['data'])


def test_five_selected_M10_payload_bytes_raw5_and_all7200_endpoint_diagnostics(monkeypatch):
    captured={};real=allocation_ui.st.download_button
    def capture(label,data,**kw):captured[kw['file_name']]=data;return real(label,data,**kw)
    monkeypatch.setattr(allocation_ui.st,'download_button',capture);app=open_view();captured.clear();widget(app.selectbox,'同一予算の母集団').set_value('lognormal').run();assert len(captured)==5
    base=allocation_ui.SMALL_EVIDENCE
    for filename,source in [('rq_m10_pooled_summary.csv','pooled_summary.csv'),('rq_m10_hypotheses.json','hypotheses.json'),('rq_m10_overview.html','figures/overview.html')]:assert captured[filename]==(base/source).read_bytes()
    rows=list(csv.DictReader(io.StringIO(captured['rq_m10_lognormal_seed0_outer0_sample.csv'].decode())));assert len(rows)==5
    with np.load(base/'originals/lognormal_seed0.npz',allow_pickle=False) as cell:
        np.testing.assert_array_equal([float(row['X']) for row in rows],cell['samples'][0]);np.testing.assert_array_equal([float(row['raw']) for row in rows],cell['raw'][0]);assert rows[0]['raw_sha256']==cell['raw_sha256'][0].decode()
    assert {r['n'] for r in rows}=={'5'} and {r['sample_t_df'] for r in rows}=={'4'} and {r['comparison_df'] for r in rows}=={'29'}
    saved=panel('lognormal');endpoints=list(csv.DictReader(io.StringIO(captured['rq_m10_lognormal_seed0_outer0_endpoints.csv'].decode())));assert len(endpoints)==7200
    for row in endpoints:
        rep,mi,k=int(row['budget_repeat_id']),list(allocation_ui.METHODS).index(row['method']),int(row['stream_id'])
        assert row['cohort']=='M10' and int(row['namespace'])==0x4D3130 and row['n']=='5' and row['alpha']=='0.05' and row['sample_t_df']=='4' and row['comparison_df']=='29'
        for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high','less_count','equal_count'):
            value,expected=float(row[key]),saved[key][0,rep,mi,k];assert value==expected or (np.isnan(value) and np.isnan(expected))
        assert (row['valid']=='True')==saved['valid'][0,rep,mi,k] and (row['tail_warning']=='True')==saved['tail_warning'][0,rep,mi,k]
        assert row['reason']==str(saved['reason_labels'][saved['reason_codes'][0,rep,mi,k]])
    assert all(isinstance(value,bytes) for value in captured.values()) and b'n5' in captured['rq_m10_overview.html'] and b'df4' in captured['rq_m10_overview.html']


@pytest.mark.parametrize('damage',['missing','raw','receipt','n','hash','diagnostics'])
def test_corrupt_or_missing_M10_stops_safely_and_old_M08_still_works(monkeypatch,tmp_path,damage):
    if damage=='missing':monkeypatch.setattr(allocation_ui,'SMALL_EVIDENCE',tmp_path)
    else:
        original_loader=allocation_ui._panel
        def altered(path,modified):
            data=original_loader(path,modified)
            if '/rq_m10/' in path:
                if damage=='raw' and '/originals/' in path:data['raw']=data['raw'].copy();data['raw'][0,0]+=.01
                if '/panels/' in path:
                    if damage=='receipt':data['original_receipt_sha256']=np.array('0'*64)
                    if damage=='n':data['n']=np.array(20)
                    if damage=='hash':data['sample_sha256']=data['sample_sha256'].copy();data['sample_sha256'][0]=b'0'*64
                    if damage=='diagnostics':data['equal_count']=data['equal_count'][...,0]
            return data
        monkeypatch.setattr(allocation_ui,'_panel',altered)
    app=open_view('M08');widget(app.selectbox,'保存コホート').set_value('M10').run();assert not app.exception and len(app.error)==1 and not app.get('plotly_chart')
    widget(app.selectbox,'保存コホート').set_value('M08').run();healthy(app)


def test_synthetic_invalid_NA_tables_render_without_zero_imputation_and_warn_valid_preserved(monkeypatch):
    loader=allocation_ui._load
    def altered(name,evidence=None):
        rows,payload=loader(name,evidence)
        if evidence==allocation_ui.SMALL_EVIDENCE:
            for row in rows:
                if row.get('method')=='bca' and row.get('distribution')=='gamma':
                    for key in ('T','A','V','M','Q','Q_over_R','lower_sd','upper_sd','S','Ase','Vse','Vbar','d','L','U','Jgap','Pgap','delta','ratio_mean_Q120_over_mean_Q30','mean_individual_Q120_Q30'):
                        if key in row:row[key]=''
                    row['complete']='False'
        return rows,payload
    real_read=Path.read_bytes
    def read_with_undefined_intervals(path):
        payload=real_read(path)
        if path==allocation_ui.SMALL_EVIDENCE/'hypotheses.json':
            document=json.loads(payload)
            for key in ('H1_gamma','H2_L_gamma','H2_U_gamma'):
                row=document['quantities'][key];row.update(valid=False,mean=None,sd=None,mcse=None,low=None,high=None,reason='incomplete_required_original_panel')
            document['population_decisions']['gamma']['H2']['mean_Q30']=None
            return json.dumps(document).encode()
        return payload
    real_panel=allocation_ui._panel
    def invalid_and_warning_panel(path,modified):
        data=real_panel(path,modified)
        if '/rq_m10/panels/gamma_seed0' in path:
            for key in ('valid','tail_warning','q_high','tau_high','reason_codes','lower','upper'):data[key]=data[key].copy()
            data['q_high'][0,0,2,0]=1;data['tau_high'][0,0,2,0]=0;data['tail_warning'][0,0,2,0]=True
            data['valid'][0,0,2,1]=False;data['lower'][0,0,2,1]=np.nan;data['upper'][0,0,2,1]=np.nan
            data['reason_labels']=np.concatenate([data['reason_labels'],np.array(['fixture_numerical_invalid'])]);data['reason_codes'][0,0,2,1]=len(data['reason_labels'])-1
        return data
    monkeypatch.setattr(Path,'read_bytes',read_with_undefined_intervals);monkeypatch.setattr(allocation_ui,'_panel',invalid_and_warning_panel)
    monkeypatch.setattr(allocation_ui,'_load',altered);app=open_view();healthy(app)
    assert widget(app.metric,'予算推定値 T=Ase').value=='未定義' and widget(app.metric,'平均V=Vbar=Vse').value=='未定義'
    assert allocation_ui._number(None)==allocation_ui._number('')=='未定義'
    for trace in plots(app)[1]['data']:assert all(v is None for v in trace['y'])
    assert any('有効警告と無効は別です' in item.value for item in app.caption)

    assert any('fixture_numerical_invalid' in item.value and '主量NA' in item.value for item in app.warning)
    diagnostics=next(item.value for item in app.dataframe if {'stream_id','q_high','equal_count'}.issubset(item.value.columns))
    assert diagnostics.iloc[0].valid and diagnostics.iloc[0].q_high==1 and diagnostics.iloc[0].tail_warning and not diagnostics.iloc[1].valid
    assert sum(len(trace['x'])//3 for trace in plots(app)[0]['data'][2:])==119
    assert len(plots(app)[6]['data'])==1 and plots(app)[6]['data'][0]['name']==allocation_ui.LABELS['lognormal']
    compared=next(item.value for item in app.dataframe if {'cohort','quantity','low995'}.issubset(item.value.columns))
    missing=compared[(compared.cohort=='M10')&(compared.distribution=='gamma')];assert len(missing)==3 and missing.low995.isna().all() and not missing.valid.any()
