"""Saved M07 UI: exact evidence, persistent originals and no numerical engine calls."""
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

from mathscape import (bootstrap,bootstrap_ui,distribution_shape,finite_k,finite_k_ui,inference,inference_ui,
                       matched_moments,matched_stability,stability)
from scripts import run_finite_k_benchmark

APP=Path(__file__).resolve().parents[1]/'app.py'
MODE='揺れの測り方にも揺れがある'


@pytest.fixture(autouse=True)
def forbid_inference_generation_and_main(monkeypatch,caplog):
    def forbidden(*args,**kwargs):raise AssertionError('saved-only UI may not infer, resample or launch experiment')
    names=('rng_for_panel','rng_for_cell','generate_sample','generate_bootstrap_means','bootstrap_means','generate_samples','draw_replicate','intervals',
           'prepare_sample','infer_single','infer_prefixes','original_metrics','replicate_metrics','t_inference','make_inputs_manifest','load_inputs','run_cell','run_benchmark','run_smoke')
    for module in (finite_k,matched_stability,matched_moments,distribution_shape,stability,bootstrap,bootstrap_ui,inference,inference_ui,run_finite_k_benchmark):
        for name in names:
            if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    monkeypatch.setattr(policies,'_shown_default_value_warning',False)
    caplog.set_level(logging.WARNING,logger=policies.__name__)
    yield
    assert not any('was created with a default value' in item.getMessage() and 'Session State API' in item.getMessage() for item in caplog.get_records('call'))


def widget(elements,label):return next(item for item in elements if item.label==label)
def healthy(app):assert not app.exception and not app.error


def open_view():
    app=AppTest.from_file(str(APP),default_timeout=30).run()
    widget(app.radio,'Labを選択').set_value('Inference Lab').run()
    widget(app.radio,'推論モード').set_value(MODE).run();healthy(app)
    return app


def plots(app):return [json.loads(item.proto.spec) for item in app.get('plotly_chart')]
def array(value):
    if isinstance(value,dict) and 'bdata' in value:return np.frombuffer(base64.b64decode(value['bdata']),dtype=value['dtype'])
    return np.asarray(value)
def original(app):return array(plots(app)[0]['data'][0]['x'])


def panel(dist='gamma',seed=0):
    with np.load(finite_k_ui.EVIDENCE/'panels'/f'{dist}_seed{seed}.npz',allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}


def test_fixed_default_and_original_exactly_match_M06_M05_hashes():
    app=open_view();saved=panel();np.testing.assert_array_equal(original(app),saved['samples'][0])
    digest=hashlib.sha256(original(app).astype('<f8').tobytes()).hexdigest()
    assert digest==saved['sample_sha256'][0].decode()
    for source in (finite_k_ui.M06,finite_k_ui.M05):
        with np.load(source/'panels/gamma_seed0.npz',allow_pickle=False) as old:
            np.testing.assert_array_equal(original(app),old['samples'][0]);assert digest==old['sample_sha256'][0].decode()
    assert any(digest in item.value and 'M06・M05保存hash' in item.value for item in app.caption)
    assert widget(app.selectbox,'測定精度のmaster seed').value==0
    assert widget(app.number_input,'測定精度の原標本ID').value==0
    assert widget(app.selectbox,'測定反復 r').value==0 and widget(app.selectbox,'比較する系列数 K').value==120
    assert widget(app.selectbox,'測定精度の手法').value=='bca'
    assert '固定α=.05・B4999・J20' in app.success[0].value
    assert plots(app)[0]['data'][0]['marker']['symbol']=='circle'


@pytest.mark.parametrize('dist',['gamma','lognormal'])
def test_K_method_r_switches_keep_same_original_and_all_endpoint_common_axis(dist):
    app=open_view();widget(app.selectbox,'測定精度の母集団').set_value(dist).run();saved=panel(dist)
    before=original(app).copy();axis=plots(app)[0]['layout']['xaxis']['range']
    values=np.concatenate([saved['samples'][0],saved['lower'][0].ravel(),saved['upper'][0].ravel()]);values=values[np.isfinite(values)]
    padding=(values.max()-values.min())*.05
    np.testing.assert_array_equal(axis,[values.min()-padding,values.max()+padding])
    widget(app.selectbox,'比較する系列数 K').set_value(30).run();widget(app.selectbox,'測定反復 r').set_value(19).run()
    widget(app.selectbox,'測定精度の手法').set_value('percentile').run();healthy(app)
    np.testing.assert_array_equal(original(app),before);assert plots(app)[0]['layout']['xaxis']['range']==axis
    trace=next(item for item in plots(app)[0]['data'] if item['name']=='percentile · 先頭30')
    actual=array(trace['x']);expected=np.array([[saved['lower'][0,19,1,k],saved['upper'][0,19,1,k],None] for k in range(30)]).ravel()
    assert actual.tolist()==expected.tolist()
    assert not any('追加90' in item['name'] for item in plots(app)[0]['data'])
    widget(app.selectbox,'比較する系列数 K').set_value(120).run()
    extra=next(item for item in plots(app)[0]['data'] if item['name']=='percentile · 追加90')
    assert len(extra['x'])==270
    widget(app.selectbox,'測定精度の手法').set_value('t').run();healthy(app)
    assert len(plots(app)[0]['data'])==2 and plots(app)[0]['layout']['xaxis']['range']==axis


def test_population_specific_selection_restore_after_mode_and_Lab_roundtrips():
    app=open_view();widget(app.selectbox,'測定精度のmaster seed').set_value(2).run();widget(app.number_input,'測定精度の原標本ID').set_value(9).run()
    widget(app.selectbox,'測定反復 r').set_value(8).run();gamma=original(app).copy()
    widget(app.selectbox,'測定精度の母集団').set_value('lognormal').run()
    assert widget(app.selectbox,'測定精度のmaster seed').value==0 and widget(app.number_input,'測定精度の原標本ID').value==0 and widget(app.selectbox,'測定反復 r').value==0
    widget(app.selectbox,'測定精度のmaster seed').set_value(1).run();widget(app.number_input,'測定精度の原標本ID').set_value(7).run();widget(app.selectbox,'測定反復 r').set_value(18).run()
    widget(app.selectbox,'比較する系列数 K').set_value(30).run();widget(app.selectbox,'測定精度の手法').set_value('percentile').run();lognormal=original(app).copy()
    assert not np.array_equal(gamma,lognormal)
    widget(app.radio,'推論モード').set_value('同じ標本でも端点は揺れる').run();widget(app.radio,'推論モード').set_value(MODE).run()
    widget(app.radio,'Labを選択').set_value('Chaos Explorer').run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();healthy(app)
    assert widget(app.radio,'推論モード').value==MODE and widget(app.selectbox,'比較する系列数 K').value==30
    assert widget(app.selectbox,'測定精度の手法').value=='percentile' and widget(app.selectbox,'測定反復 r').value==18
    np.testing.assert_array_equal(original(app),lognormal)
    widget(app.selectbox,'測定精度の母集団').set_value('gamma').run()
    assert widget(app.selectbox,'測定精度のmaster seed').value==2 and widget(app.number_input,'測定精度の原標本ID').value==9 and widget(app.selectbox,'測定反復 r').value==8
    np.testing.assert_array_equal(original(app),gamma)


def test_selected_metrics_and_twenty_paired_Ase_values_match_saved_rows():
    app=open_view();rows=pd.read_csv(finite_k_ui.EVIDENCE/'K_metrics.csv.gz');current=rows[(rows.distribution=='gamma')&(rows.master_seed==0)&(rows.outer_id==0)&(rows.replicate_id==0)&(rows.method=='bca')&(rows.K==120)].iloc[0]
    for label,key in [('下端点SD','lower_sd'),('上端点SD','upper_sd'),('合成変動 S','S'),('標本SE（s/√20）','se'),('測定Ase','Ase'),('測定Apop','Apop')]:
        assert widget(app.metric,label).value==format(current[key],'.6g')
    reps=pd.read_csv(finite_k_ui.EVIDENCE/'replicate_metrics.csv.gz');selected=reps[(reps.distribution=='gamma')&(reps.master_seed==0)&(reps.outer_id==0)&(reps.method=='bca')]
    for trace,k in zip(plots(app)[1]['data'],(30,120)):
        np.testing.assert_array_equal(array(trace['x']),np.arange(20));np.testing.assert_allclose(array(trace['y']),selected[f'A{k}'],rtol=1e-13)
    groups=pd.read_csv(finite_k_ui.EVIDENCE/'group_metrics.csv.gz');expected=groups[(groups.distribution=='gamma')&(groups.master_seed==0)&(groups.outer_id==0)&(groups.replicate_id==0)&(groups.method=='bca')]
    for field in ['lower_mean','upper_mean','lower_var','upper_var','V','A']:
        np.testing.assert_allclose(app.dataframe[1].value[field],expected[field],rtol=1e-13)
    first=selected.iloc[0];gap=plots(app)[2]
    np.testing.assert_allclose(array(gap['data'][0]['y']),[first.Jgap,first.Pgap,first.Rgap],rtol=1e-13)
    assert gap['data'][1]['y']==[first.delta]
    assert app.dataframe[2].value.algebra_ok.tolist()==[True]


def test_full_pooled12_seed36_decomposition24_exact_saved_tables_and_counts():
    app=open_view()
    for index,name,count in [(4,'pooled_summary.csv',12),(5,'summary.csv',36),(6,'decomposition_summary.csv',24)]:
        expected=pd.read_csv(finite_k_ui.EVIDENCE/name);actual=app.dataframe[index].value
        assert len(actual)==count
        pd.testing.assert_frame_equal(actual.astype(object).where(actual.notna(),None),expected.astype(object).where(expected.notna(),None),check_dtype=False,rtol=1e-12,atol=1e-14)
    pooled=app.dataframe[4].value
    assert set(pooled.original_total)=={30} and set(pooled.loc[pooled.method=='t','interval_count'])=={30}
    assert set(pooled.loc[(pooled.method!='t')&(pooled.K==120),'interval_count'])=={30*20*120}
    assert set(pooled.loc[(pooled.method!='t')&(pooled.K==30),'interval_count'])=={30*20*30}
    for trace,dist in zip(plots(app)[3]['data'],finite_k_ui.LABELS):
        rows=pooled[(pooled.distribution==dist)&(pooled.method=='bca')].set_index('K').loc[[30,120]]
        np.testing.assert_allclose(array(trace['y']),rows.A,rtol=1e-13)
        np.testing.assert_allclose(array(trace['error_y']['array']),rows.A_decision_high-rows.A,rtol=1e-12)
    assert any('比の平均' in item.value or '原標本ごとの比の平均' in item.value for item in app.caption)
    assert any('内側乱数だけの厳密区間ではありません' in item.value for item in app.info)


def test_four_confirmatory_intervals_exact_band_variance_units_and_joint_status():
    app=open_view();saved=json.loads((finite_k_ui.EVIDENCE/'hypotheses.json').read_text())
    for chart,H in zip(plots(app)[4:6],('H1','H2')):
        for trace,dist in zip(chart['data'],finite_k_ui.LABELS):
            row=saved['quantities'][f'{H}_{dist}'];assert row['N']==30
            assert trace['x']==[row['mean']] and trace['error_x']['array']==[row['high']-row['mean']] and trace['error_x']['arrayminus']==[row['mean']-row['low']]
    band=next(shape for shape in plots(app)[4]['layout']['shapes'] if shape['type']=='rect');assert band['x0']==-.0025 and band['x1']==.0025
    assert any('被覆率0.25ppではありません' in item.value and 'SDの50%低減ではありません' in item.value for item in app.caption)
    assert any('H1全体：支持' in item.value for item in app.markdown) and any('H2全体：支持' in item.value for item in app.markdown)
    assert any('科学仮説の支持ではありません' in item.value for item in app.caption)


def test_old_K1_NA_and_old_results_bytes_unchanged():
    path=finite_k_ui.M05/'pooled_summary.csv';before=path.read_bytes();app=open_view()
    old=app.dataframe[-1].value;assert set(old.K)=={1}
    assert old[['lower_sd','upper_sd','S','Ase','Apop']].isna().all().all()
    local=app.dataframe[0].value;assert (local.loc[local.method=='t',['Q30','Q120','d']]==0).all().all()
    assert path.read_bytes()==before


def test_free_alpha_is_independent_and_restores_after_saved_mode():
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run()
    widget(app.selectbox,'有意水準 α').set_value(.10).run();widget(app.radio,'推論モード').set_value(MODE).run();healthy(app)
    assert '固定α=.05・B4999' in app.success[0].value
    widget(app.radio,'推論モード').set_value('t推論の探索').run();assert widget(app.selectbox,'有意水準 α').value==.10


def test_all_download_payloads_exact_saved_bytes_and_sample_endpoint_roundtrip(monkeypatch):
    captured={};real=finite_k_ui.st.download_button
    def capture(label,data,**kwargs):captured[kwargs['file_name']]=data;return real(label,data,**kwargs)
    monkeypatch.setattr(finite_k_ui.st,'download_button',capture);app=open_view()
    assert len(captured)==5 and all(isinstance(value,bytes) and value for value in captured.values())
    for file,path in [('rq_m07_pooled_summary.csv','pooled_summary.csv'),('rq_m07_hypotheses.json','hypotheses.json'),('rq_m07_overview.html','figures/overview.html')]:
        assert captured[file]==(finite_k_ui.EVIDENCE/path).read_bytes()
    assert b'<html>' in captured['rq_m07_overview.html'] and b'plotly' in captured['rq_m07_overview.html'].lower()
    rows=list(csv.DictReader(io.StringIO(captured['rq_m07_gamma_seed0_outer0_sample.csv'].decode())));values=np.array([float(row['X']) for row in rows],dtype='<f8')
    assert len(rows)==20;np.testing.assert_array_equal(values,original(app));assert hashlib.sha256(values.tobytes()).hexdigest()==rows[0]['sample_sha256']
    endpoints=list(csv.DictReader(io.StringIO(captured['rq_m07_gamma_seed0_outer0_endpoints.csv'].decode())));assert len(endpoints)==7200;saved=panel()
    for row in endpoints:
        r,mi,k=int(row['replicate_id']),list(finite_k_ui.METHODS).index(row['method']),int(row['inner_id'])
        for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high'):
            actual,expected=float(row[key]),saved[key][0,r,mi,k]
            assert actual==expected or (np.isnan(actual) and np.isnan(expected))
        assert (row['valid']=='True')==saved['valid'][0,r,mi,k] and (row['tail_warning']=='True')==saved['tail_warning'][0,r,mi,k]


@pytest.mark.parametrize('failure',['missing','hash'])
def test_missing_evidence_or_provenance_mismatch_stops_without_computing(monkeypatch,tmp_path,failure):
    if failure=='missing':monkeypatch.setattr(finite_k_ui,'EVIDENCE',tmp_path)
    else:monkeypatch.setattr(finite_k_ui,'_hashes',lambda *args:np.full(10000,'0'*64))
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();widget(app.radio,'推論モード').set_value(MODE).run()
    assert not app.exception and len(app.error)==1 and not app.get('plotly_chart')


@pytest.mark.parametrize('dist',['gamma','lognormal'])
def test_t_control_zero_points_have_no_undefined_MC_error_bars(dist):
    app=open_view();widget(app.selectbox,'測定精度の母集団').set_value(dist).run();widget(app.selectbox,'測定精度の手法').set_value('t').run();healthy(app)
    for trace in plots(app)[3]['data']:
        np.testing.assert_array_equal(array(trace['y']),[0,0]);assert 'error_y' not in trace
    assert any('tの99%比較区間はSE0でNA' in item.value for item in app.caption)
    for trace in plots(app)[1]['data']:np.testing.assert_array_equal(array(trace['y']),np.zeros(20))


def test_old_M06_settings_restore_after_new_view_and_new_settings_remain_independent():
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run()
    widget(app.radio,'推論モード').set_value('同じ標本でも端点は揺れる').run()
    widget(app.number_input,'端点変動の原標本ID').set_value(43).run();widget(app.selectbox,'端点変動のbootstrap回数 B').set_value(999).run();widget(app.selectbox,'端点変動の内側系列 k').set_value(29).run()
    old=original(app).copy();widget(app.radio,'推論モード').set_value(MODE).run()
    widget(app.number_input,'測定精度の原標本ID').set_value(8).run();widget(app.selectbox,'測定反復 r').set_value(15).run();new=original(app).copy()
    widget(app.radio,'推論モード').set_value('同じ標本でも端点は揺れる').run();healthy(app)
    assert widget(app.number_input,'端点変動の原標本ID').value==43 and widget(app.selectbox,'端点変動のbootstrap回数 B').value==999 and widget(app.selectbox,'端点変動の内側系列 k').value==29
    np.testing.assert_array_equal(original(app),old)
    widget(app.radio,'推論モード').set_value(MODE).run();healthy(app)
    assert widget(app.number_input,'測定精度の原標本ID').value==8 and widget(app.selectbox,'測定反復 r').value==15
    np.testing.assert_array_equal(original(app),new)


def test_sample_and_endpoint_payloads_keep_full_saved_original_when_K_r_method_change():
    saved=panel('lognormal',2);state=dict(distribution='lognormal',K=30,method='percentile');selection=dict(seed=2,outer=9,r=19)
    digest=saved['sample_sha256'][9].decode();sample=finite_k_ui._sample_csv(saved['samples'][9],state,selection,digest)
    endpoints=finite_k_ui._endpoint_csv(saved,state,selection,digest)
    state.update(K=120,method='t');selection['r']=0
    assert sample==finite_k_ui._sample_csv(saved['samples'][9],state,selection,digest)
    assert endpoints==finite_k_ui._endpoint_csv(saved,state,selection,digest)
    rows=list(csv.DictReader(io.StringIO(endpoints.decode())));assert len(rows)==7200
    assert {row['outer_id'] for row in rows}=={'9'} and {row['master_seed'] for row in rows}=={'2'} and {row['sample_sha256'] for row in rows}=={digest}


def test_visible_endpoint_diagnostics_and_warning_counts_use_saved_arrays():
    app=open_view();saved=panel();diagnostic=app.dataframe[-2].value
    assert len(diagnostic)==120 and set(diagnostic.method)=={'bca'} and set(diagnostic.replicate_id)=={0}
    for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high'):
        np.testing.assert_array_equal(diagnostic[key],saved[key][0,0,2])
    np.testing.assert_array_equal(diagnostic.valid,saved['valid'][0,0,2]);np.testing.assert_array_equal(diagnostic.tail_warning,saved['tail_warning'][0,0,2])
    count=int(saved['tail_warning'][0,0,2].sum())
    assert any(f'tau<10警告{count}' in item.value for item in app.caption)
    assert not any(item.label in ('有意水準 α','bootstrap回数 B') for item in app.selectbox)
