"""Saved-only M11 navigation, scientific gates and lossless export regressions."""
import csv
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pandas  # Import NumPy's Generator type annotations before RNG patching.
import pytest
from streamlit.testing.v1 import AppTest

from mathscape import enumeration_ui as view
from mathscape import enumerated_reference, small_sample
from scripts import analyze_enumerated_reference, audit_enumerated_reference

APP = Path(__file__).resolve().parents[1] / 'app.py'


@pytest.fixture(autouse=True)
def saved_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('M11 presentation cannot run science or random generation')
    for module, names in ((enumerated_reference, ('ordered_indices','enumerate_distribution','inverse_cdf','sample_context','build_reference','stream_comparison','endpoint_metrics','classify_original','group_decision')),
                          (small_sample, ('generate_cohort','draw_original','generate_bootstrap_means','infer_single')),
                          (analyze_enumerated_reference, ('analyze','analyze_records')),
                          (audit_enumerated_reference, ('audit',))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for name in ('default_rng','Generator','PCG64','SeedSequence'):
        monkeypatch.setattr(np.random, name, forbidden)


def widget(elements, label):
    return next(item for item in elements if item.label == label)


def open_view():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state['inference_mode'] = '同じ予算で、どう測る？'
    app.run()
    widget(app.radio, 'Labを選択').set_value('Inference Lab').run()
    widget(app.selectbox, '保存コホート').set_value('M10').run()
    assert not app.exception and not app.error
    return app


def test_m10_preview_preserves_legacy_counts_then_detail_has_five_exports(monkeypatch):
    payloads = {}
    original = view.st.download_button
    def capture(label, data, **kwargs):
        payloads[kwargs['file_name']] = data
        return original(label, data, **kwargs)
    monkeypatch.setattr(view.st, 'download_button', capture)
    app = open_view()
    assert len(app.get('download_button')) == 5
    assert any('21/30' in item.value and '22/30' in item.value for item in app.info)
    payloads.clear()
    widget(app.selectbox, 'M10の保存詳細').set_value('全列挙参照とのずれ（M11）').run()
    assert not app.exception and not app.error
    assert len(app.get('download_button')) == len(payloads) == 5
    assert widget(app.metric, 'G：標準化ガンマ · BCa 0.10SE以内').value == '21 / 30'
    assert widget(app.metric, 'L：標準化対数正規 · BCa 0.10SE以内').value == '22 / 30'
    assert payloads['rq_m11_summary.csv'] == (view.EVIDENCE/'summary.csv').read_bytes()
    assert payloads['rq_m11_hypotheses.json'] == (view.EVIDENCE/'hypotheses.json').read_bytes()
    assert payloads['rq_m11_overview.html'] == (view.EVIDENCE/'figures/overview.html').read_bytes()
    assert len(list(csv.DictReader(io.StringIO(payloads['rq_m11_gamma_seed0_outer0_comparison.csv'].decode())))) == 2400


def test_methods_population_ID_and_legacy_allocation_state_are_retained():
    app = open_view()
    widget(app.selectbox, '同一予算の手法').set_value('t').run()
    widget(app.selectbox, '同一予算の配分').set_value('P60').run()
    widget(app.number_input, '同一予算の原標本ID').set_value(7).run()
    widget(app.selectbox, '予算反復 r').set_value(13).run()
    widget(app.selectbox, 'M10の保存詳細').set_value('全列挙参照とのずれ（M11）').run()
    widget(app.selectbox, '全列挙詳細の手法').set_value('percentile').run()
    widget(app.selectbox, '同一予算の母集団').set_value('lognormal').run()
    widget(app.selectbox, '同一予算のmaster seed').set_value(2).run()
    widget(app.number_input, '同一予算の原標本ID').set_value(9).run()
    widget(app.selectbox, '同一予算の母集団').set_value('gamma').run()
    assert widget(app.number_input, '同一予算の原標本ID').value == 7
    assert widget(app.selectbox, '予算反復 r').value == 13
    assert widget(app.selectbox, '同一予算の手法').value == 't'
    assert widget(app.selectbox, '同一予算の配分').value == 'P60'
    assert not app.exception and not app.error
    widget(app.selectbox, 'M10の保存詳細').set_value('配分の保存結果').run()
    assert not app.exception and not app.error and len(app.get('download_button')) == 5
    assert widget(app.selectbox, '同一予算の手法').value == 't'


def test_fixed_representative_button_restores_saved_identity():
    app = open_view()
    widget(app.selectbox, 'M10の保存詳細').set_value('全列挙参照とのずれ（M11）').run()
    widget(app.selectbox, '同一予算の母集団').set_value('lognormal').run()
    widget(app.selectbox, '予算反復 r').set_value(7).run()
    widget(app.selectbox, '同一予算の母集団').set_value('gamma').run()
    widget(app.selectbox, '予算反復 r').set_value(13).run()
    example = json.loads((view.EVIDENCE/'representatives.json').read_text())['examples'][-1]
    next(button for button in app.button if button.key == 'enum_example_'+example['distribution']+example['selection']).click().run()
    assert not app.exception and not app.error
    assert widget(app.selectbox, '同一予算の母集団').value == example['distribution']
    assert widget(app.selectbox, '同一予算のmaster seed').value == example['master_seed']
    assert widget(app.number_input, '同一予算の原標本ID').value == example['outer_id']
    assert widget(app.selectbox, '予算反復 r').value == 7


@pytest.mark.parametrize('distribution', ['gamma', 'lognormal'])
def test_every_exported_numeric_field_matches_saved_arrays_losslessly(distribution):
    data = view.load_verified(distribution=distribution, seed=2, outer=9)
    ref_bytes, comparison_bytes = view.reference_csv(data), view.comparison_csv(data)
    assert view.reference_csv(data) == ref_bytes and view.comparison_csv(data) == comparison_bytes
    reference = list(csv.DictReader(io.StringIO(ref_bytes.decode())))
    enumerated = [row for row in reference if row['record_type']=='enumeration']
    support = [row for row in reference if row['record_type']=='support']
    endpoints = [row for row in reference if row['record_type']=='reference']
    assert len(enumerated)==3125 and len(endpoints)==4
    for i, row in enumerate(enumerated):
        assert float(row['mean']) == data['reference']['means'][i]
        assert [int(row[f'index_{k}']) for k in range(5)] == data['reference']['indices'][i].tolist()
    for i, row in enumerate(support):
        assert float(row['support']) == data['reference']['support'][i]
        assert int(row['frequency']) == data['reference']['frequency'][i]
        assert int(row['cumulative']) == data['reference']['cumulative'][i]
    for row in endpoints:
        mi, ei = view.METHODS.index(row['method']), view.ENDPOINTS.index(row['endpoint'])
        for key, values in data['reference'].items():
            if values.shape == (2,2):
                if values.dtype.kind == 'f':
                    assert (row[key]=='NA' and not np.isfinite(values[mi,ei])) or float(row[key])==values[mi,ei]
                elif values.dtype.kind in 'iu':
                    assert int(row[key]) == values[mi,ei]
                else:
                    assert row[key] == str(values[mi,ei])
    comparison = list(csv.DictReader(io.StringIO(comparison_bytes.decode())))
    assert len(comparison)==2400
    for key, values in data['comparison'].items():
        if values.shape not in ((2400,), (2400,2), (2400,2,2)):
            continue
        for coordinate in np.ndindex(values.shape[1:]):
            name = key if not coordinate else f'{view.METHODS[coordinate[0]]}_'+(f'{view.ENDPOINTS[coordinate[1]]}_' if len(coordinate)==2 else '')+key
            for i, row in enumerate(comparison):
                expected = values[(i,)+coordinate]
                if values.dtype.kind=='f':
                    assert (row[name]=='NA' and not np.isfinite(expected)) or float(row[name])==expected
                else:
                    assert row[name] == str(expected)


@pytest.mark.parametrize('damage', ['audit', 'json', 'selected_npz', 'main'])
def test_failed_scientific_gate_or_changed_saved_bytes_cannot_be_presented(tmp_path, damage):
    evidence = tmp_path/'rq_m11'
    shutil.copytree(view.EVIDENCE, evidence)
    if damage in ('audit','main'):
        path = evidence/('independent_audit.json' if damage=='audit' else 'main_resources.json')
        payload=json.loads(path.read_text());payload['status']='stopped';path.write_text(json.dumps(payload))
    elif damage=='json':
        with (evidence/'hypotheses.json').open('ab') as handle:handle.write(b' ')
    else:
        with (evidence/'references/gamma_seed0_outer0.npz').open('ab') as handle:handle.write(b'x')
    with pytest.raises(ValueError):view.load_verified(evidence=evidence)


def test_html_scope_and_once_only_writer(tmp_path):
    html=(view.EVIDENCE/'figures/overview.html').read_text()
    assert '21/30' in html and '22/30' in html and '全144000系列の図示ではありません' in html
    assert '<script src=' not in html
    existing=tmp_path/'existing.html';existing.write_text('keep')
    with pytest.raises(FileExistsError):view.write_overview(existing)
    assert existing.read_text()=='keep'


def test_synthetic_export_preserves_invalid_reason_and_valid_endpoint_warning():
    data=view.load_verified()
    data['reference']={key:value.copy() for key,value in data['reference'].items()}
    data['comparison']={key:value.copy() for key,value in data['comparison'].items()}
    data['reference']['valid'][0]=False
    data['reference']['reason'][0]='fixture_reference_invalid'
    data['reference']['E_cdf'][0,0]=np.nan
    data['comparison']['saved_valid'][0,0]=False
    data['comparison']['saved_reason'][0,0]='fixture_invalid'
    data['comparison']['saved_endpoint'][0,0]=np.nan
    data['comparison']['saved_valid'][1,0]=True
    data['comparison']['saved_warning'][1,0]=True
    data['comparison']['saved_q'][1,0,1]=1.0
    data['comparison']['saved_tau_high'][1,0]=0.0
    reference=list(csv.DictReader(io.StringIO(view.reference_csv(data).decode())))
    invalid=next(row for row in reference if row['record_type']=='reference' and row['method']=='bca' and row['endpoint']=='lower')
    assert invalid['valid']=='False' and invalid['reason']=='fixture_reference_invalid' and invalid['E_cdf']=='NA'
    comparison=list(csv.DictReader(io.StringIO(view.comparison_csv(data).decode())))
    assert comparison[0]['bca_saved_reason']=='fixture_invalid'
    assert comparison[0]['bca_lower_saved_endpoint']=='NA'
    assert comparison[1]['bca_saved_valid']=='True' and comparison[1]['bca_saved_warning']=='True'
    assert comparison[1]['bca_upper_saved_q']=='1.0' and comparison[1]['bca_saved_tau_high']=='0.0'


def test_unverified_detail_shows_hold_without_exports_or_scientific_plot(monkeypatch):
    app=open_view()
    def broken(*args,**kwargs):raise ValueError('fixture audit mismatch')
    monkeypatch.setattr(view,'load_verified',broken)
    widget(app.selectbox,'M10の保存詳細').set_value('全列挙参照とのずれ（M11）').run()
    assert not app.exception and len(app.error)==1
    assert '未検証・保留' in app.error[0].value
    assert not app.get('download_button') and not app.get('plotly_chart')
