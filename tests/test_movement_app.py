"""M12 saved presentation: no scientific calls, precise points, state and exports."""
import base64
import csv
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mathscape import movement_ui as view
from mathscape import probability_movement,enumerated_reference,small_sample
from scripts import (analyze_probability_movement,audit_probability_movement,run_probability_movement,
                     analyze_enumerated_reference,audit_enumerated_reference)

APP=Path(__file__).resolve().parents[1]/'app.py'
MODE='同じ予算で、どう測る？'
M11_DETAIL='全列挙参照とのずれ（M11）'
M12_DETAIL='補正確率を替えた端点移動（M12）'


@pytest.fixture(autouse=True)
def saved_only(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('saved M12 view must not run science or RNG')
    modules=(probability_movement,enumerated_reference,small_sample,analyze_probability_movement,
             audit_probability_movement,run_probability_movement,analyze_enumerated_reference,audit_enumerated_reference)
    names=('analyze','analyze_records','analyze_sample','map_original','select_rank','verify_distribution','endpoint_metrics',
           'classify_original','group_decision','population_decision','joint_status','cumulative_index','validate_inputs',
           'endpoint_summary','classifications','representatives','rank_representatives','classification_rows','summary_rows',
           'audit','worker','supervise','manifest','generate_cohort','draw_original','generate_bootstrap_means','infer_single',
           'ordered_indices','enumerate_distribution','inverse_cdf','sample_context','build_reference','stream_comparison','reference_intervals')
    for module in modules:
        for name in names:
            if hasattr(module,name):monkeypatch.setattr(module,name,forbidden)
    for name in ('default_rng','Generator','PCG64','SeedSequence'):monkeypatch.setattr(np.random,name,forbidden)


def widget(elements,label):return next(item for item in elements if item.label==label)
def healthy(app):assert not app.exception and not app.error

def open_view():
    app=AppTest.from_file(str(APP),default_timeout=30)
    app.session_state['inference_mode']=MODE
    app.run();widget(app.radio,'Labを選択').set_value('Inference Lab').run()
    widget(app.selectbox,'保存コホート').set_value('M10').run()
    widget(app.selectbox,'M10の保存詳細').set_value(M11_DETAIL).run()
    widget(app.selectbox,'M11の保存解析').set_value(M12_DETAIL).run()
    healthy(app);return app


def plots(app):return [json.loads(item.proto.spec) for item in app.get('plotly_chart')]
def array(value):
    if isinstance(value,dict) and 'bdata' in value:return np.frombuffer(base64.b64decode(value['bdata']),dtype=value['dtype'])
    return np.asarray(value)


def test_initial_saved_scope_status_and_exact_five_payloads(monkeypatch):
    captured={};original=view.st.download_button
    def capture(label,data,**kwargs):captured[kwargs['file_name']]=data;return original(label,data,**kwargs)
    monkeypatch.setattr(view.st,'download_button',capture)
    app=open_view();healthy(app);captured.clear();app.run();assert len(captured)==5
    assert len(widget(app.radio,'Labを選択').options)==4 and len(widget(app.radio,'推論モード').options)==8
    assert len(app.get('download_button'))==5
    assert any('確率の揺れは、端点をどれだけ動かす？' in item.value for item in app.subheader)
    for file,source in [('rq_m12_original_metrics.csv','original_metrics.csv'),('rq_m12_endpoint_metrics.csv','endpoint_metrics.csv'),
                        ('rq_m12_summary.csv','summary.csv'),('rq_m12_decision.json','decision.json')]:
        assert captured[file]==(view.EVIDENCE/source).read_bytes()
    data=view.load_verified()
    assert captured['rq_m12_gamma_seed0_outer0_series.csv']==view.series_csv(data)
    decision=json.loads(captured['rq_m12_decision.json']);assert decision['groups']['gamma']['pass_count']==26 and decision['groups']['lognormal']['pass_count']==25
    assert decision['status']=='not_supported'
    assert widget(app.metric,'G：標準化ガンマ · M12 0.10SE以内').value=='26 / 30'
    assert widget(app.metric,'L：標準化対数正規 · M12 0.10SE以内').value=='25 / 30'
    assert any('記述的不支持' in item.value for item in app.info)


@pytest.mark.parametrize('distribution',['gamma','lognormal'])
def test_five_exports_repeatably_match_exact_source_and_all_selected_series_values(distribution):
    bundle=view.load_verified(distribution=distribution,seed=2,outer=9)
    outputs=view.exports(bundle);assert len(outputs)==5
    expected_names={'rq_m12_original_metrics.csv','rq_m12_endpoint_metrics.csv','rq_m12_summary.csv',
                    f'rq_m12_{distribution}_seed2_outer9_series.csv','rq_m12_decision.json'}
    assert {name for _,_,name,_ in outputs}==expected_names
    for _,payload,name,_ in outputs:
        assert isinstance(payload,bytes) and payload
        if name.endswith('_series.csv'):assert payload==view.series_csv(bundle)
        else:
            source={'rq_m12_original_metrics.csv':'original_metrics.csv','rq_m12_endpoint_metrics.csv':'endpoint_metrics.csv',
                    'rq_m12_summary.csv':'summary.csv','rq_m12_decision.json':'decision.json'}[name]
            assert payload==(view.EVIDENCE/source).read_bytes()
    payload=view.series_csv(bundle);assert payload==view.series_csv(bundle)
    rows=list(csv.DictReader(io.StringIO(payload.decode())));assert len(rows)==2400
    arrays=bundle['data']
    for index,row in enumerate(rows):
        assert row['distribution']==distribution and int(row['master_seed'])==2 and int(row['outer_id'])==9
        assert row['sample_sha256']==bundle['digest'] and row['n']=='5' and row['B']=='4999'
        assert float(row['alpha'])==.05
        for key,values in arrays.items():
            if values.shape not in ((2400,),(2400,2),(2,)) or key in ('endpoints',):continue
            if values.shape==(2400,):coordinates=[(key,values[index])]
            else:
                coordinates=[(endpoint+'_'+key,values[index,ei] if values.shape==(2400,2) else values[ei]) for ei,endpoint in enumerate(('lower','upper'))]
            for name,expected in coordinates:
                if values.dtype.kind=='f':assert row[name]=='NA' and not np.isfinite(expected) or float(row[name])==expected
                else:assert row[name]==str(expected)


@pytest.mark.parametrize('damage',['audit','main','json','selected','source'])
def test_unaudited_or_modified_bytes_are_rejected_before_scientific_display(tmp_path,damage):
    evidence=tmp_path/'rq_m12';shutil.copytree(view.EVIDENCE,evidence)
    if damage in ('audit','main'):
        path=evidence/('independent_audit.json' if damage=='audit' else 'main_resources.json')
        saved=json.loads(path.read_bytes());saved['status']='stopped';path.write_text(json.dumps(saved))
    elif damage=='json':
        with (evidence/'decision.json').open('ab') as handle:handle.write(b' ')
    elif damage=='selected':
        with (evidence/'series/gamma_seed0_outer0.npz').open('ab') as handle:handle.write(b' ')
    else:
        saved=json.loads((evidence/'independent_audit.json').read_bytes())
        saved['source_sha256']['evidence/rq_m11/references/gamma_seed0_outer0.npz']='0'*64
        (evidence/'independent_audit.json').write_text(json.dumps(saved))
    with pytest.raises((ValueError,OSError)):view.load_verified(evidence=evidence)


def test_unaudited_UI_is_hold_without_download_or_plot(monkeypatch):
    app=open_view()
    def broken(*args,**kwargs):raise ValueError('fixture audit mismatch')
    monkeypatch.setattr(view,'load_verified',broken);app.run()
    assert not app.exception and len(app.error)==1 and '未検証' in app.error[0].value and '保留' in app.error[0].value
    assert not app.get('plotly_chart') and not app.get('download_button')


def test_population_original_r_and_dedicated_controls_restore_with_M11_Lab_modes():
    app=open_view()
    widget(app.selectbox,'同一予算の手法').set_value('t').run();widget(app.selectbox,'同一予算の配分').set_value('P60').run()
    widget(app.selectbox,'同一予算のmaster seed').set_value(1).run();widget(app.number_input,'同一予算の原標本ID').set_value(7).run();widget(app.selectbox,'予算反復 r').set_value(13).run()
    widget(app.selectbox,'移動を表示する端点').set_value('upper').run();widget(app.number_input,'移動を表示する系列 k').set_value(119).run()
    widget(app.selectbox,'同一予算の母集団').set_value('lognormal').run();widget(app.selectbox,'同一予算のmaster seed').set_value(2).run();widget(app.number_input,'同一予算の原標本ID').set_value(9).run();widget(app.selectbox,'予算反復 r').set_value(7).run()
    widget(app.selectbox,'移動を表示する端点').set_value('upper').run();widget(app.number_input,'移動を表示する系列 k').set_value(29).run()
    widget(app.selectbox,'M11の保存解析').set_value('全列挙端点との総ずれ（M11）').run()
    widget(app.selectbox,'全列挙詳細の手法').set_value('percentile').run()
    assert len(app.get('download_button'))==5
    widget(app.selectbox,'M11の保存解析').set_value(M12_DETAIL).run()
    assert widget(app.selectbox,'移動を表示する端点').value=='upper' and widget(app.number_input,'移動を表示する系列 k').value==29
    widget(app.radio,'推論モード').set_value('揺れの測り方にも揺れがある').run();widget(app.radio,'推論モード').set_value(MODE).run()
    widget(app.radio,'Labを選択').set_value('Probability Explorer').run();widget(app.radio,'Labを選択').set_value('Inference Lab').run();healthy(app)
    assert widget(app.selectbox,'M11の保存解析').value==M12_DETAIL
    assert widget(app.selectbox,'移動を表示する端点').value=='upper' and widget(app.number_input,'移動を表示する系列 k').value==29
    assert widget(app.number_input,'同一予算の原標本ID').value==9 and widget(app.selectbox,'予算反復 r').value==7
    widget(app.selectbox,'同一予算の母集団').set_value('gamma').run();healthy(app)
    assert widget(app.selectbox,'同一予算のmaster seed').value==1 and widget(app.number_input,'同一予算の原標本ID').value==7 and widget(app.selectbox,'予算反復 r').value==13
    assert widget(app.selectbox,'移動を表示する端点').value=='upper' and widget(app.number_input,'移動を表示する系列 k').value==119
    assert widget(app.selectbox,'同一予算の手法').value=='t' and widget(app.selectbox,'同一予算の配分').value=='P60'
    bundle=view.load_verified(distribution='gamma',seed=1,outer=7)
    assert any(bundle['digest'] in item.value for item in app.caption)
    widget(app.selectbox,'M11の保存解析').set_value('全列挙端点との総ずれ（M11）').run()
    assert widget(app.selectbox,'全列挙詳細の手法').value=='percentile'


def test_fixed_representatives_restore_IDs_without_reselecting_r_or_methods():
    app=open_view();widget(app.selectbox,'予算反復 r').set_value(13).run()
    widget(app.selectbox,'同一予算の母集団').set_value('lognormal').run();widget(app.selectbox,'予算反復 r').set_value(7).run()
    examples=json.loads((view.EVIDENCE/'representatives.json').read_bytes())['examples']
    for example in examples:
        next(button for button in app.button if button.key=='movement_example_'+example['distribution']+example['selection']).click().run();healthy(app)
        assert widget(app.selectbox,'同一予算の母集団').value==example['distribution']
        assert widget(app.selectbox,'同一予算のmaster seed').value==example['master_seed'] and widget(app.number_input,'同一予算の原標本ID').value==example['outer_id']
        assert widget(app.selectbox,'予算反復 r').value==(13 if example['distribution']=='gamma' else 7)


@pytest.mark.parametrize('distribution',['gamma','lognormal'])
@pytest.mark.parametrize('endpoint',['lower','upper'])
def test_exact_saved_cdf_markers_selection_metrics_and_tables(distribution,endpoint):
    app=open_view();widget(app.selectbox,'同一予算の母集団').set_value(distribution).run()
    widget(app.selectbox,'移動を表示する端点').set_value(endpoint).run();widget(app.selectbox,'予算反復 r').set_value(19).run();widget(app.number_input,'移動を表示する系列 k').set_value(119).run();healthy(app)
    bundle=view.load_verified(distribution=distribution);data=bundle['data'];ei=('lower','upper').index(endpoint);index=2399
    chart=plots(app)[0];np.testing.assert_array_equal(array(chart['data'][0]['x']),data['support']);np.testing.assert_array_equal(array(chart['data'][0]['y']),data['cdf_probability'])
    for trace,x,y in [(chart['data'][1],data['A'][ei],data['q_star'][ei]),(chart['data'][2],data['C'][index,ei],data['q_B'][index,ei]),(chart['data'][3],data['Y'][index,ei],data['q_B'][index,ei])]:
        assert trace['x']==[x] and trace['y']==[y]
    selected=app.dataframe[0].value
    for ei2 in range(2):
        for key in ('q_star','A','A_index'):assert str(selected.iloc[ei2][key])==str(data[key][ei2])
        for key in ('q_B','q_delta','C','Y','selected_index','support_step','g','h','d','valid','movement_code','q_changed','q_changed_same','endpoint_identity_residual','source_tail_count'):
            observed=selected.iloc[ei2][key];expected=data[key][index,ei2]
            if isinstance(expected,(float,np.floating)):assert observed=='NA' and not np.isfinite(expected) or float(observed)==expected
            else:assert str(observed)==str(expected)
    row=next(row for row in bundle['original_metrics'] if row['distribution']==distribution and row['master_seed']=='0' and row['outer_id']=='0')
    for label,key in [('選択原標本の u（SE単位）','u'),('下端の移動RMS（SE単位）','lower_r'),('上端の移動RMS（SE単位）','upper_r')]:assert widget(app.metric,label).value==format(float(row[key]),'.17g')
    for index2,name in [(2,'summary'),(3,'original_metrics'),(4,'endpoint_metrics'),(5,'classification'),(6,'classification_ids')]:
        expected=pd.DataFrame(bundle[name]);pd.testing.assert_frame_equal(app.dataframe[index2].value,expected,check_dtype=False)
    assert any('因果的寄与率' in item.value and '独立な成分' in item.value for item in app.markdown)
    assert any('36' in item.value and '143964' in item.value for item in app.caption)


def test_nearby_distinct_points_and_invalid_warning_render_from_saved_classifications(monkeypatch):
    bundle=view.load_verified();data=bundle['data'];tiny=np.nextafter(1.,2.)
    data['support']=np.array([0.,1.,tiny,2.]);data['cdf_probability']=np.array([.1,.5,.8,1.])
    data['A'][0]=1.;data['q_star'][0]=.4;data['A_index'][0]=1
    data['C'][0,0]=tiny;data['Y'][0,0]=tiny;data['q_B'][0,0]=.6;data['q_delta'][0,0]=.2
    data['g'][0,0]=tiny-1.;data['h'][0,0]=0.;data['d'][0,0]=tiny-1.
    data['valid'][0,0]=True;data['selected_index'][0,0]=2;data['support_step'][0,0]=1
    data['movement_code'][0,0]=1;data['q_changed'][0,0]=True;data['q_changed_same'][0,0]=False
    data['source_warning'][0]=True;data['source_tail_count'][0,0]=0.
    # A separate retained invalid slot must remain undefined, with its reason.
    data['reason_labels']=np.r_[data['reason_labels'],np.array(['fixture_invalid_endpoint'])]
    code=len(data['reason_labels'])-1
    data['valid'][1,0]=False;data['reason_code'][1,0]=code;data['movement_code'][1,0]=-2
    for key in ('g','h','d','C','q_delta'):data[key][1,0]=np.nan
    monkeypatch.setattr(view,'load_verified',lambda *args,**kwargs:bundle)
    app=open_view();healthy(app)
    row=app.dataframe[0].value.iloc[0];assert float(row.C)==tiny and float(row.A)==1. and row.movement_code==1
    assert float(row.g)>0 and row.valid and row.source_warning
    assert any('上へ移動' in item.value for item in app.markdown)
    csv_rows=list(csv.DictReader(io.StringIO(view.series_csv(bundle).decode())))
    assert csv_rows[0]['lower_C']==repr(float(tiny)) and float(csv_rows[0]['lower_g'])>0 and csv_rows[0]['lower_valid']=='True'
    assert csv_rows[1]['lower_reason']=='fixture_invalid_endpoint' and csv_rows[1]['lower_g']=='NA' and csv_rows[1]['lower_valid']=='False'
    original=next(row for row in bundle['original_metrics'] if row['distribution']=='gamma' and row['master_seed']=='0' and row['outer_id']=='0')
    original.update(complete='False',u='NA',margin='NA',pass_flag='NA',lower_r='NA',invalid='True',reason='fixture_invalid_endpoint')
    bundle['decision']['groups']['gamma'].update(complete_N=29,invalid_N=1,unknown_N=1,pass_count=25,status='inconclusive')
    widget(app.number_input,'移動を表示する系列 k').set_value(1).run();healthy(app)
    row=app.dataframe[0].value.iloc[0];assert row.g=='NA' and not row.valid and row.reason=='fixture_invalid_endpoint'
    assert any('未定義' in item.value for item in app.markdown)
    assert widget(app.metric,'選択原標本の u（SE単位）').value=='未定義' and widget(app.metric,'下端の移動RMS（SE単位）').value=='未定義'
    assert widget(app.metric,'G：標準化ガンマ · M12 0.10SE以内').value=='25 / 30'
    assert any('必要無効1' in item.value and '判定不明1' in item.value and '計画分母は30' in item.value for item in app.caption)


def test_two_sessions_keep_dedicated_state_and_evidence_constants_independent():
    first=open_view();widget(first.selectbox,'移動を表示する端点').set_value('upper').run();widget(first.number_input,'移動を表示する系列 k').set_value(41).run()
    second=open_view();widget(second.selectbox,'同一予算の母集団').set_value('lognormal').run();widget(second.number_input,'同一予算の原標本ID').set_value(9).run();widget(second.number_input,'移動を表示する系列 k').set_value(19).run()
    first.run();healthy(first)
    assert widget(first.selectbox,'同一予算の母集団').value=='gamma' and widget(first.number_input,'同一予算の原標本ID').value==0
    assert widget(first.selectbox,'移動を表示する端点').value=='upper' and widget(first.number_input,'移動を表示する系列 k').value==41
    second.run();healthy(second)
    assert widget(second.selectbox,'同一予算の母集団').value=='lognormal' and widget(second.number_input,'同一予算の原標本ID').value==9
    assert widget(second.selectbox,'移動を表示する端点').value=='lower' and widget(second.number_input,'移動を表示する系列 k').value==19
    assert view.EVIDENCE==Path(__file__).resolve().parents[1]/'evidence/rq_m12'
