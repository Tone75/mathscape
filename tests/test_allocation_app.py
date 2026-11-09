"""Saved allocation UI: exact evidence, state, payloads and no inference calls."""
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

from mathscape import (allocation,allocation_ui,bootstrap,bootstrap_ui,distribution_shape,finite_k,inference,inference_ui,
                       matched_moments,matched_stability,stability)
from scripts import run_allocation_benchmark

APP=Path(__file__).resolve().parents[1]/'app.py'
MODE='同じ予算で、どう測る？'


@pytest.fixture(autouse=True)
def forbid_inference_resampling_and_experiments(monkeypatch,caplog):
    def forbidden(*args,**kwargs):raise AssertionError('saved allocation view must only read/plot evidence')
    names=('rng_for_panel','rng_for_cell','generate_sample','generate_bootstrap_means','bootstrap_means','generate_samples','draw_replicate','intervals','prepare_sample','infer_single','infer_prefixes',
           'original_metrics','replicate_metrics','budget_metrics','t_inference','make_inputs_manifest','load_inputs','run_cell','run_benchmark','run_smoke')
    for module in (allocation,finite_k,matched_stability,matched_moments,distribution_shape,stability,bootstrap,bootstrap_ui,inference,inference_ui,run_allocation_benchmark):
        for name in names:
            if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden);monkeypatch.setattr(policies,'_shown_default_value_warning',False)
    caplog.set_level(logging.WARNING,logger=policies.__name__)
    yield
    assert not any('was created with a default value' in record.getMessage() and 'Session State API' in record.getMessage() for record in caplog.get_records('call'))


def widget(elements,label):return next(item for item in elements if item.label==label)
def healthy(app):assert not app.exception and not app.error


def open_view():
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();widget(app.radio,'推論モード').set_value(MODE).run();healthy(app)
    return app


def plots(app):return [json.loads(item.proto.spec) for item in app.get('plotly_chart')]
def array(value):
    if isinstance(value,dict) and 'bdata' in value:return np.frombuffer(base64.b64decode(value['bdata']),dtype=value['dtype'])
    return np.asarray(value)
def original(app):return array(plots(app)[0]['data'][0]['x'])


def panel(dist='gamma',seed=0):
    with np.load(allocation_ui.EVIDENCE/'panels'/f'{dist}_seed{seed}.npz',allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}


def test_initial_fixed_scope_and_original_match_all_three_accepted_sources():
    app=open_view();saved=panel();np.testing.assert_array_equal(original(app),saved['samples'][0]);digest=hashlib.sha256(original(app).astype('<f8').tobytes()).hexdigest()
    assert digest==saved['sample_sha256'][0].decode()
    for source in (allocation_ui.M07,allocation_ui.M06,allocation_ui.M05):
        with np.load(source/'panels/gamma_seed0.npz',allow_pickle=False) as old:
            np.testing.assert_array_equal(original(app),old['samples'][0]);assert digest==old['sample_sha256'][0].decode()
    assert any(digest in item.value and 'M07・M06・M05保存hash' in item.value for item in app.caption)
    assert widget(app.selectbox,'同一予算のmaster seed').value==0 and widget(app.number_input,'同一予算の原標本ID').value==0 and widget(app.selectbox,'予算反復 r').value==0
    assert widget(app.selectbox,'同一予算の配分').value=='P30' and widget(app.selectbox,'同一予算の手法').value=='bca'
    assert '固定α=.05・B4999・R20' in app.success[0].value
    assert plots(app)[0]['data'][0]['marker']['symbol']=='circle'


@pytest.mark.parametrize('dist',['gamma','lognormal'])
def test_every_allocation_method_and_repeat_keeps_original_and_full_common_axis(dist):
    app=open_view();widget(app.selectbox,'同一予算の母集団').set_value(dist).run();saved=panel(dist);before=original(app).copy();axis=plots(app)[0]['layout']['xaxis']['range']
    allx=np.r_[saved['samples'][0],saved['lower'][0].ravel(),saved['upper'][0].ravel()];allx=allx[np.isfinite(allx)];pad=(allx.max()-allx.min())*.05
    np.testing.assert_array_equal(axis,[allx.min()-pad,allx.max()+pad])
    widget(app.selectbox,'予算反復 r').set_value(19).run();widget(app.selectbox,'同一予算の手法').set_value('percentile').run()
    for name,(j,k) in allocation_ui.ALLOCATIONS.items():
        widget(app.selectbox,'同一予算の配分').set_value(name).run();healthy(app)
        np.testing.assert_array_equal(original(app),before);assert plots(app)[0]['layout']['xaxis']['range']==axis
        traces=plots(app)[0]['data'][2:];assert len(traces)==j
        for group,trace in enumerate(traces):
            expected=np.array([[saved['lower'][0,19,1,stream],saved['upper'][0,19,1,stream],None] for stream in range(group*k,(group+1)*k)]).ravel()
            assert array(trace['x']).tolist()==expected.tolist()
        assert sum(len(trace['x'])//3 for trace in traces)==120
    widget(app.selectbox,'同一予算の手法').set_value('t').run();healthy(app)
    assert len(plots(app)[0]['data'])==2 and plots(app)[0]['layout']['xaxis']['range']==axis


def test_population_state_restore_and_mode_Lab_roundtrips_keep_hash():
    app=open_view();widget(app.selectbox,'同一予算のmaster seed').set_value(2).run();widget(app.number_input,'同一予算の原標本ID').set_value(9).run();widget(app.selectbox,'予算反復 r').set_value(7).run();gamma=original(app).copy()
    widget(app.selectbox,'同一予算の母集団').set_value('lognormal').run()
    assert widget(app.selectbox,'同一予算のmaster seed').value==0 and widget(app.number_input,'同一予算の原標本ID').value==0 and widget(app.selectbox,'予算反復 r').value==0
    widget(app.selectbox,'同一予算のmaster seed').set_value(1).run();widget(app.number_input,'同一予算の原標本ID').set_value(8).run();widget(app.selectbox,'予算反復 r').set_value(18).run()
    widget(app.selectbox,'同一予算の配分').set_value('P60').run();widget(app.selectbox,'同一予算の手法').set_value('percentile').run();lognormal=original(app).copy();assert not np.array_equal(gamma,lognormal)
    widget(app.radio,'推論モード').set_value('揺れの測り方にも揺れがある').run();widget(app.radio,'推論モード').set_value(MODE).run()
    widget(app.radio,'Labを選択').set_value('Chaos Explorer').run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();healthy(app)
    assert widget(app.radio,'推論モード').value==MODE and widget(app.selectbox,'予算反復 r').value==18 and widget(app.selectbox,'同一予算の配分').value=='P60' and widget(app.selectbox,'同一予算の手法').value=='percentile'
    np.testing.assert_array_equal(original(app),lognormal)
    widget(app.selectbox,'同一予算の母集団').set_value('gamma').run();assert widget(app.selectbox,'同一予算のmaster seed').value==2 and widget(app.number_input,'同一予算の原標本ID').value==9 and widget(app.selectbox,'予算反復 r').value==7
    np.testing.assert_array_equal(original(app),gamma)


def test_six_metrics_R20_T_all_allocations_seven_groups_and_decomposition_exact():
    app=open_view();budgets=pd.read_csv(allocation_ui.EVIDENCE/'budget_metrics.csv.gz');rows=budgets[(budgets.distribution=='gamma')&(budgets.master_seed==0)&(budgets.outer_id==0)&(budgets.method=='bca')]
    current=rows[(rows.budget_repeat_id==0)&(rows.allocation=='P30')].iloc[0]
    for label,key in [('組平均の下端点SD','lower_sd'),('組平均の上端点SD','upper_sd'),('組平均のS','S'),('標本SE（s/√20）','se'),('予算推定値 T=Ase','T'),('平均V=Vbar=Vse','Vbar')]:assert widget(app.metric,label).value==format(current[key],'.6g')
    assert current.Vse==current.Vbar and current.Vse!=pytest.approx(current.Ase**2,rel=1e-5)
    for trace,name in zip(plots(app)[1]['data'],allocation_ui.ALLOCATIONS):
        np.testing.assert_array_equal(array(trace['x']),np.arange(20));np.testing.assert_allclose(array(trace['y']),rows.loc[rows.allocation==name,'T'],rtol=1e-13)
    groups=pd.read_csv(allocation_ui.EVIDENCE/'group_metrics.csv.gz');selected=groups[(groups.distribution=='gamma')&(groups.master_seed==0)&(groups.outer_id==0)&(groups.method=='bca')&(groups.budget_repeat_id==0)]
    assert len(app.dataframe[2].value)==7
    for key in ('group_start','group_stop','lower_var','upper_var','V','A'):np.testing.assert_allclose(app.dataframe[2].value[key],selected[key],rtol=1e-13)
    for trace,name in zip(plots(app)[2]['data'],allocation_ui.ALLOCATIONS):np.testing.assert_allclose(array(trace['y']),selected.loc[selected.allocation==name,'A'],rtol=1e-13)
    decomp=pd.read_csv(allocation_ui.EVIDENCE/'decomposition_metrics.csv.gz');row=decomp[(decomp.distribution=='gamma')&(decomp.master_seed==0)&(decomp.outer_id==0)&(decomp.method=='bca')&(decomp.budget_repeat_id==0)].iloc[0]
    np.testing.assert_allclose(array(plots(app)[3]['data'][0]['y']),[row.Jgap,row.Pgap],rtol=1e-13);assert plots(app)[3]['data'][1]['y']==[row.delta]
    assert app.dataframe[3].value.algebra_ok.tolist()==[True] and not any('Rgap' in item['name'] for item in plots(app)[3]['data'])
    assert any('Vse≠Ase²' in item.value for item in app.caption)


def test_full18_54_24_tables_and_pooled_M_Q_bars_exact_and_N_not_RJK():
    app=open_view()
    for index,name,count in [(5,'pooled_summary.csv',18),(6,'summary.csv',54),(7,'decomposition_summary.csv',24)]:
        expected=pd.read_csv(allocation_ui.EVIDENCE/name);actual=app.dataframe[index].value;assert len(actual)==count
        pd.testing.assert_frame_equal(actual.astype(object).where(actual.notna(),None),expected.astype(object).where(expected.notna(),None),check_dtype=False,rtol=1e-12,atol=1e-14)
    pooled=app.dataframe[5].value;assert set(pooled.original_total)=={30} and set(pooled.loc[pooled.method=='t','interval_count'])=={30}
    assert set(pooled.loc[pooled.method!='t','interval_count'])=={30*20*120}
    for fig,field in zip(plots(app)[4:6],('M','Q')):
        for trace,dist in zip(fig['data'],allocation_ui.LABELS):
            rows=pooled[(pooled.distribution==dist)&(pooled.method=='bca')].set_index('allocation').loc[list(allocation_ui.ALLOCATIONS)]
            np.testing.assert_allclose(array(trace['y']),rows[field],rtol=1e-13);np.testing.assert_allclose(array(trace['error_y']['array']),rows[field+'_decision_high']-rows[field],rtol=1e-12,atol=1e-14)
    assert any('R/J/KをNに足しません' in item.value for item in app.info)


def test_all_six_995_intervals_H1_band_H2_zero_and_descriptive_ratio_no_ratioCI():
    app=open_view();saved=json.loads((allocation_ui.EVIDENCE/'hypotheses.json').read_text())
    for trace,dist in zip(plots(app)[6]['data'],allocation_ui.LABELS):
        row=saved['quantities']['H1_'+dist];assert trace['x']==[row['mean']] and trace['error_x']['arrayminus']==[row['mean']-row['low']] and trace['error_x']['array']==[row['high']-row['mean']]
    band=next(shape for shape in plots(app)[6]['layout']['shapes'] if shape['type']=='rect');assert band['x0']==-.0025 and band['x1']==.0025
    for trace,(dist,name) in zip(plots(app)[7]['data'],[(d,h) for d in allocation_ui.LABELS for h in ('H2_L','H2_U')]):
        row=saved['quantities'][name+'_'+dist];assert row['N']==30 and trace['x']==[row['mean']] and trace['error_x']['array']==[row['high']-row['mean']] and trace['error_x']['arrayminus']==[row['mean']-row['low']]
    ratio=plots(app)[8];band=next(shape for shape in ratio['layout']['shapes'] if shape['type']=='rect');assert band['x0']==.8 and band['x1']==1.25
    decomp=pd.read_csv(allocation_ui.EVIDENCE/'decomposition_summary.csv')
    for trace,dist in zip(ratio['data'],allocation_ui.LABELS):
        row=decomp[(decomp.distribution==dist)&(decomp.master_seed=='pooled')&(decomp.method=='bca')].iloc[0]
        assert trace['x'][0]==pytest.approx(row.ratio_mean_Q120_over_mean_Q30,rel=1e-13) and 'error_x' not in trace
    assert any('H1全体：支持' in item.value for item in app.markdown) and any('H2全体：支持' in item.value for item in app.markdown)
    assert any('「±25%」やSDの帯ではありません' in item.value and '比の点に信頼区間を付けておらず' in item.value for item in app.caption)


@pytest.mark.parametrize('dist',['gamma','lognormal'])
def test_t_control_zero_points_comparison_SE0_NA_no_errorbars(dist):
    app=open_view();widget(app.selectbox,'同一予算の母集団').set_value(dist).run();widget(app.selectbox,'同一予算の手法').set_value('t').run();healthy(app)
    for fig in plots(app)[4:6]:
        for trace in fig['data']:np.testing.assert_array_equal(array(trace['y']),np.zeros(3));assert 'error_y' not in trace
    for trace in plots(app)[1]['data']:np.testing.assert_array_equal(array(trace['y']),np.zeros(20))
    assert any('SE0の比較区間はNA' in item.value for item in app.caption)


def test_old_M05_K1_NA_and_existing_saved_evidence_not_changed():
    path=allocation_ui.M05/'pooled_summary.csv';before=path.read_bytes();app=open_view();old=app.dataframe[-1].value
    assert set(old.K)=={1} and old[['lower_sd','upper_sd','S','Ase','Apop']].isna().all().all()
    local=app.dataframe[0].value;assert (local.loc[local.method=='t',['M','Q','Q_over_R']]==0).all().all() and path.read_bytes()==before


def test_old_free_alpha_and_M07_parameters_restore_independently():
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();widget(app.selectbox,'有意水準 α').set_value(.10).run()
    widget(app.radio,'推論モード').set_value('揺れの測り方にも揺れがある').run();widget(app.number_input,'測定精度の原標本ID').set_value(7).run();widget(app.selectbox,'測定反復 r').set_value(15).run();widget(app.selectbox,'比較する系列数 K').set_value(30).run();old=original(app).copy()
    widget(app.radio,'推論モード').set_value(MODE).run();assert '固定α=.05・B4999・R20' in app.success[0].value
    widget(app.number_input,'同一予算の原標本ID').set_value(8).run();widget(app.selectbox,'予算反復 r').set_value(19).run();new=original(app).copy()
    widget(app.radio,'推論モード').set_value('揺れの測り方にも揺れがある').run();assert widget(app.number_input,'測定精度の原標本ID').value==7 and widget(app.selectbox,'測定反復 r').value==15 and widget(app.selectbox,'比較する系列数 K').value==30;np.testing.assert_array_equal(original(app),old)
    widget(app.radio,'推論モード').set_value('t推論の探索').run();assert widget(app.selectbox,'有意水準 α').value==.10
    widget(app.radio,'推論モード').set_value(MODE).run();healthy(app);assert widget(app.number_input,'同一予算の原標本ID').value==8 and widget(app.selectbox,'予算反復 r').value==19;np.testing.assert_array_equal(original(app),new)


def test_all_five_payloads_exact_sourcebytes_full_sample_and7200_all_diags(monkeypatch):
    captured={};real=allocation_ui.st.download_button
    def capture(label,data,**kwargs):captured[kwargs['file_name']]=data;return real(label,data,**kwargs)
    monkeypatch.setattr(allocation_ui.st,'download_button',capture);app=open_view()
    assert len(captured)==5 and all(isinstance(payload,bytes) and payload for payload in captured.values())
    for name,path in [('rq_m08_pooled_summary.csv','pooled_summary.csv'),('rq_m08_hypotheses.json','hypotheses.json'),('rq_m08_overview.html','figures/overview.html')]:assert captured[name]==(allocation_ui.EVIDENCE/path).read_bytes()
    assert b'<html>' in captured['rq_m08_overview.html'] and b'plotly' in captured['rq_m08_overview.html'].lower()
    sample=list(csv.DictReader(io.StringIO(captured['rq_m08_gamma_seed0_outer0_sample.csv'].decode())));values=np.array([float(row['X']) for row in sample],dtype='<f8')
    assert len(sample)==20;np.testing.assert_array_equal(values,original(app));assert hashlib.sha256(values.tobytes()).hexdigest()==sample[0]['sample_sha256']
    endpoints=list(csv.DictReader(io.StringIO(captured['rq_m08_gamma_seed0_outer0_endpoints.csv'].decode())));saved=panel();assert len(endpoints)==7200
    for row in endpoints:
        r,mi,k=int(row['budget_repeat_id']),list(allocation_ui.METHODS).index(row['method']),int(row['stream_id'])
        for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high'):
            actual,expected=float(row[key]),saved[key][0,r,mi,k];assert actual==expected or (np.isnan(actual) and np.isnan(expected))
        assert (row['valid']=='True')==saved['valid'][0,r,mi,k] and (row['tail_warning']=='True')==saved['tail_warning'][0,r,mi,k]


def test_visible_diagnostics_warning_counts_and_payload_independent_of_allocation_r_method():
    app=open_view();saved=panel();table=app.dataframe[-2].value;assert len(table)==120
    for key in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high'):np.testing.assert_array_equal(table[key],saved[key][0,0,2])
    assert table.allocation_group.tolist()==[k//30 for k in range(120)]
    assert any(f'tau<10警告{int(saved["tail_warning"][0,0,2].sum())}' in item.value for item in app.caption)
    state=dict(distribution='lognormal',allocation='P30',method='bca');selection=dict(seed=2,outer=9,r=19);saved=panel('lognormal',2);digest=saved['sample_sha256'][9].decode()
    sample=allocation_ui._sample_csv(saved['samples'][9],state,selection,digest);endpoints=allocation_ui._endpoint_csv(saved,state,selection,digest)
    state.update(allocation='P120',method='t');selection['r']=0
    assert sample==allocation_ui._sample_csv(saved['samples'][9],state,selection,digest) and endpoints==allocation_ui._endpoint_csv(saved,state,selection,digest)


@pytest.mark.parametrize('failure',['missing','hash'])
def test_missing_input_or_lineage_mismatch_graceful_without_charts(monkeypatch,tmp_path,failure):
    if failure=='missing':monkeypatch.setattr(allocation_ui,'EVIDENCE',tmp_path)
    else:monkeypatch.setattr(allocation_ui,'_hashes',lambda *args:np.full(10000,'0'*64))
    app=AppTest.from_file(str(APP),default_timeout=30).run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();widget(app.radio,'推論モード').set_value(MODE).run()
    assert not app.exception and len(app.error)==1 and not app.get('plotly_chart')
