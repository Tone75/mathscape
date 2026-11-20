"""M11 saved-only presentation and lossless exports; no scientific imports."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/rq_m11'
METHODS = ('bca', 'percentile')
ENDPOINTS = ('lower', 'upper')
LABELS = {'gamma': 'G：標準化ガンマ', 'lognormal': 'L：標準化対数正規'}
COLORS = {'gamma': '#FFB86B', 'lognormal': '#46DCC0'}
STATUS = {'supported': '記述的支持', 'not_supported': '記述的不支持', 'inconclusive': '保留'}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    return json.loads(path.read_bytes())


def _npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def _rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8'))))


def load_verified(evidence=EVIDENCE, root=ROOT, distribution='gamma', seed=0, outer=0):
    """Verify saved execution/audit receipts and consumed bytes before display."""
    evidence, root = Path(evidence), Path(root)
    if distribution not in LABELS or seed not in (0, 1, 2) or outer not in range(10):
        raise ValueError('selected identity outside fixed saved design')
    audit = _json(evidence / 'independent_audit.json')
    main = _json(evidence / 'main_resources.json')
    correction = evidence / 'audit_correction_resources.json'
    audited = _json(correction if correction.exists() else evidence / 'audit_resources.json')
    receipt = _json(evidence / 'input_receipt.json')
    manifest = _json(evidence / 'manifest.json')
    if (audit.get('status') != 'passed' or audit.get('mismatch_count') != 0 or
        audit.get('question_id') != 'RQ-M11' or
        any(item.get('status') != 'completed' or item.get('exit_code') != 0 for item in (main, audited))):
        raise ValueError('main and successful independent audit must both be complete')
    commit = receipt['preanalysis_commit']
    if any(item.get('preanalysis_commit') != commit for item in (main, audited, audit)):
        raise ValueError('execution and scientific audit commits differ')
    if (receipt['manifest_sha256'] != _digest(evidence / 'manifest.json') or
        receipt['inputs'] != manifest['known_inputs'] or
        receipt['originals'] != manifest['original_ids_and_receipt']):
        raise ValueError('frozen input receipt differs')
    for name in ('hypotheses.json', 'representatives.json', 'analysis_resources.json', 'input_receipt.json'):
        if audit['output_json_sha256'].get(name) != _digest(evidence / name):
            raise ValueError('audited JSON differs: ' + name)
    tables = ('summary.csv', 'original_metrics.csv', 'endpoint_metrics.csv', 'reference_metrics.csv')
    for name in tables:
        if audit['output_npz_and_csv_sha256'].get(name) != _digest(evidence / name):
            raise ValueError('audited table differs: ' + name)
    stem = f'{distribution}_seed{seed}_outer{outer}.npz'
    for folder in ('references', 'comparisons'):
        relative = folder + '/' + stem
        if audit['output_npz_and_csv_sha256'].get(relative) != _digest(evidence / relative):
            raise ValueError('audited selected arrays differ: ' + relative)
    source = root / 'evidence/rq_m10'
    panel_relative = f'panels/{distribution}_seed{seed}.npz'
    panel_path = source / panel_relative
    if audit['input_panel_and_stream_sha256'].get(panel_relative) != _digest(panel_path):
        raise ValueError('audited selected M10 panel differs')
    for relative in (f'evidence/rq_m10/{panel_relative}',
                     f'evidence/rq_m10/originals/{distribution}_seed{seed}.npz',
                     'evidence/rq_m10/originals/receipt.json'):
        if receipt['inputs'][relative]['sha256'] != _digest(root / relative):
            raise ValueError('accepted selected M10 bytes differ')
    panel = _npz(panel_path)
    reference, comparison = (_npz(evidence / folder / stem) for folder in ('references', 'comparisons'))
    sample = panel['samples'][outer]
    digest = hashlib.sha256(np.asarray(sample, dtype='<f8').tobytes()).hexdigest()
    for saved in (reference, comparison):
        if (str(saved['sample_sha256']) != digest or str(saved['distribution']) != distribution or
            int(saved['master_seed']) != seed or int(saved['outer_id']) != outer or
            tuple(saved['methods'].astype(str)) != METHODS or
            tuple(saved['endpoints'].astype(str)) != ENDPOINTS or int(saved['n']) != 5):
            raise ValueError('selected sample / array axes differ')
    if (panel['sample_sha256'][outer].decode() != digest or
        reference['indices'].shape != (3125, 5) or comparison['saved_endpoint'].shape != (2400, 2, 2)):
        raise ValueError('selected saved dimensions differ')
    result = {name.removesuffix('.csv'): _rows(evidence / name) for name in tables}
    result.update(reference=reference, comparison=comparison, sample=sample,
        hypotheses=_json(evidence / 'hypotheses.json'), representatives=_json(evidence / 'representatives.json'),
        audit=audit, evidence=evidence, distribution=distribution, seed=seed, outer=outer, digest=digest)
    return result


def preview():
    try:
        data = load_verified()
        groups = data['hypotheses']['groups']
        return '全列挙参照とのずれ（M11）：BCaは G ' + str(groups['gamma']['pass_count']) + '/30、L ' + str(groups['lognormal']['pass_count']) + '/30 が0.10SE以内。判定：' + STATUS[data['hypotheses']['status']] + '。詳細を選ぶと保存端点とのずれを探索できます。'
    except (OSError, ValueError, KeyError, TypeError):
        return '全列挙参照とのずれ（M11）：保存結果は未検証・保留です。詳細で検証状態を確認できます。'


def _cell(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return repr(value) if np.isfinite(value) else 'NA'
    return value


def _csv(records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fields)
    writer.writeheader()
    writer.writerows({key: _cell(value) for key, value in record.items()} for record in records)
    return buffer.getvalue().encode('utf-8')


def _metadata(data):
    return dict(distribution=data['distribution'], master_seed=data['seed'], outer_id=data['outer'],
        sample_sha256=data['digest'], n=5, B=4999, R=20, K=120,
        reference_convention='binary64 ordered enumeration / inverse CDF exact integer selection')


def reference_csv(data):
    saved, metadata = data['reference'], _metadata(data)
    records = []
    for index, (indices, mean) in enumerate(zip(saved['indices'], saved['means'])):
        records.append(dict(metadata, record_type='enumeration', ordered_row=index,
            **{f'index_{i}': value for i, value in enumerate(indices)}, mean=mean))
    for index, (support, frequency, cumulative) in enumerate(zip(saved['support'], saved['frequency'], saved['cumulative'])):
        records.append(dict(metadata, record_type='support', support_index=index,
            support=support, frequency=frequency, cumulative=cumulative))
    for mi, method in enumerate(METHODS):
        for ei, endpoint in enumerate(ENDPOINTS):
            row = dict(metadata, record_type='reference', method=method, endpoint=endpoint)
            for key, value in saved.items():
                if value.shape == (2, 2):
                    row[key] = value[mi, ei]
                elif key in ('valid', 'reason'):
                    row[key] = value[mi]
                elif key in ('theta', 'se', 'a', 'enum_less_count', 'enum_equal_count', 'p_star', 'z0_star', 'enum_size'):
                    row[key] = value.item()
            records.append(row)
    return _csv(records)


def comparison_csv(data):
    saved, metadata = data['comparison'], _metadata(data)
    records = []
    for index in range(2400):
        row = dict(metadata, series_id=index)
        for key, value in saved.items():
            if value.shape == (2400,):
                row[key] = value[index]
            elif value.shape == (2400, 2):
                for mi, method in enumerate(METHODS):
                    row[f'{method}_{key}'] = value[index, mi]
            elif value.shape == (2400, 2, 2):
                for mi, method in enumerate(METHODS):
                    for ei, endpoint in enumerate(ENDPOINTS):
                        row[f'{method}_{endpoint}_{key}'] = value[index, mi, ei]
        records.append(row)
    return _csv(records)


def _selected_rows(data, table, method):
    return [row for row in data[table] if row['distribution'] == data['distribution'] and
            int(row['master_seed']) == data['seed'] and int(row['outer_id']) == data['outer'] and row['method'] == method]


def endpoint_figure(data, method, endpoint):
    mi, ei = METHODS.index(method), ENDPOINTS.index(endpoint)
    fig = go.Figure(go.Histogram(x=data['comparison']['saved_endpoint'][:, mi, ei],
        name='保存2400系列', marker_color=COLORS[data['distribution']], nbinsx=45))
    for field, label, dash, color in (('E_cdf', '逆CDF参照', 'solid', '#FFFFFF'),
                                     ('E_linear', 'linear参照', 'dash', '#A8A2FF')):
        value = float(data['reference'][field][mi, ei])
        if np.isfinite(value):
            fig.add_vline(x=value, line_dash=dash, line_color=color, annotation_text=label)
    fig.update_layout(template='plotly_dark', height=300, title=f'{endpoint} · {method}',
        xaxis_title='計算端点（原標本の観測単位）', yaxis_title='保存系列数', showlegend=False)
    return fig


def _choose_example(state, example):
    state['distribution'] = example['distribution']
    selection = state['selections'][example['distribution']]
    selection.update(seed=int(example['master_seed']), outer=int(example['outer_id']))
    st.session_state['au_distribution'] = example['distribution']
    st.session_state['au_seed'] = selection['seed']
    st.session_state['au_outer'] = selection['outer']
    st.session_state['au_r'] = selection['r']


def render_enumeration_view(state):
    selection = state['selections'][state['distribution']]
    st.subheader('全通りを数えた基準から、どれだけずれる？')
    try:
        data = load_verified(distribution=state['distribution'], seed=selection['seed'], outer=selection['outer'])
    except (OSError, KeyError, ValueError, IndexError, TypeError) as error:
        st.error(f'科学判定は未検証・保留です。保存対応を確認できません（{type(error).__name__}）。')
        return
    st.session_state.setdefault('enumeration_detail',state.get('enumeration_detail','全列挙端点との総ずれ（M11）'))
    detail=st.selectbox('M11の保存解析', ['全列挙端点との総ずれ（M11）','補正確率を替えた端点移動（M12）'], key='enumeration_detail')
    state['enumeration_detail']=detail
    if detail=='補正確率を替えた端点移動（M12）':
        from mathscape.movement_ui import render_movement_view
        render_movement_view(state)
        return
    groups = data['hypotheses']['groups']
    for column, distribution in zip(st.columns(2), LABELS):
        group = groups[distribution]
        column.metric(LABELS[distribution] + ' · BCa 0.10SE以内', f'{group["pass_count"]} / {group["planned_N"]}')
    st.info('保存判定：' + STATUS[data['hypotheses']['status']] + '。全30個が完備し27/30以上が基準です。各原標本で上下端点RMSEの大きい方 e≤0.10 を合格とします。母集団の90%保証やM10保留の原因説明ではありません。')
    st.caption(f'監査済み保存結果 · {LABELS[data["distribution"]]} · seed{data["seed"]} · ID{data["outer"]} · SHA-256 {data["digest"]}。新しい原標本・再標本・推論は生成しません。')
    st.session_state.setdefault('enumeration_method',state.get('enumeration_method','bca'))
    method = st.selectbox('全列挙詳細の手法', METHODS, format_func=lambda name:'BCa（主対象）' if name=='bca' else 'percentile（記述対照）', key='enumeration_method')
    state['enumeration_method']=method
    original = _selected_rows(data, 'original_metrics', method)[0]
    st.metric('選択原標本の e（標本SE単位）', original['e'])
    st.caption('合格=' + original['pass_flag'] + ' ／ 余裕 0.10−e=' + original['margin'] + '。表示丸めや数値許容差で判定を変えません。')
    columns = st.columns(2)
    for column, endpoint in zip(columns, ENDPOINTS):
        column.plotly_chart(endpoint_figure(data, method, endpoint), width='stretch', config={'displaylogo':False})
    endpoint_rows = _selected_rows(data, 'endpoint_metrics', method)
    fields = ('endpoint','b','sd_ddof0','rmse','mean_absolute_error','abs_p50','abs_p95','abs_max','valid_series','invalid_series','missing_series')
    st.dataframe([{key: row[key] for key in fields} for row in endpoint_rows], hide_index=True, width='stretch')
    st.caption('bは保存端点と逆CDF参照の平均差、sqrt(v)はddof0の変動、RMSEは両方を含みます。2400系列はR20×120の共通系列で、3配分を重複計上しません。')
    with st.expander('固定した最小／最大e例に移動'):
        for example in data['representatives']['examples']:
            st.button(f'{LABELS[example["distribution"]]} · {example["selection"]} · seed{example["master_seed"]}/ID{example["outer_id"]}',
                key='enum_example_'+example['distribution']+example['selection'], on_click=_choose_example, args=(state, example))
    with st.expander('参照規約・tie・CDF境界と規約差'):
        st.write('参照はn5の順序付き添字3125通りを、保存binary64・M10平均演算順で評価した条件付き有限分布です。実数真値や必ず到達するB→∞極限ではありません。支持点は整数頻度を持ち、qのbinary64整数比と累積整数頻度を比較して逆CDFを選びます。linear参照も別に保存しています。')
        st.write('保存端点−逆CDF参照 =（保存端点−linear参照）+（linear参照−逆CDF参照）。MSEには交差項があり、前者を純粋なモンテカルロ成分や因果寄与とは呼びません。')
        st.dataframe(_selected_rows(data, 'reference_metrics', method), hide_index=True, width='stretch')
        st.dataframe(endpoint_rows, hide_index=True, width='stretch')
        st.caption('有限Bのtau=B×min(q,1−q)と、列挙参照の3125×min(q*,1−q*)を区別します。q=0/1やtail<10は有効なら警告で、除外しません。半tieには厳密な<と==を使い、iscloseを使いません。')
    with st.expander('全60原標本・両方法の保存指標と群要約'):
        st.dataframe(data['summary'], hide_index=True, width='stretch')
        st.dataframe(data['original_metrics'], hide_index=True, width='stretch')
        st.dataframe(data['endpoint_metrics'], hide_index=True, width='stretch')
    st.caption('主解析完了・独立監査一致を確認済み。監査は全60個の列挙／全保存誤差を照合し、有限Bの生成過程を再構成したのは既存36添字系列のみです。M08/M09とpoolせず、M10のH1/H2保留は更新しません。')
    prefix=f'rq_m11_{data["distribution"]}_seed{data["seed"]}_outer{data["outer"]}'
    for label, payload, filename, mime in (
        ('M11保存要約CSVを保存', (EVIDENCE/'summary.csv').read_bytes(), 'rq_m11_summary.csv', 'text/csv'),
        ('M11保存判定JSONを保存', (EVIDENCE/'hypotheses.json').read_bytes(), 'rq_m11_hypotheses.json', 'application/json'),
        ('選択標本の全列挙参照CSVを保存', reference_csv(data), prefix+'_reference.csv', 'text/csv'),
        ('選択標本の2400系列比較CSVを保存', comparison_csv(data), prefix+'_comparison.csv', 'text/csv'),
        ('M11自己完結HTMLを保存', (EVIDENCE/'figures/overview.html').read_bytes(), 'rq_m11_overview.html', 'text/html')):
        st.download_button(label, payload, file_name=filename, mime=mime, key='enumeration_download_'+mime+'_'+filename)


def write_overview(path, evidence=EVIDENCE, root=ROOT):
    """Publish once from verified saved arrays; never calls a science function."""
    path = Path(path)
    data = load_verified(evidence, root)
    figures = []
    fig = go.Figure()
    for distribution in LABELS:
        rows = [row for row in data['original_metrics'] if row['distribution']==distribution and row['method']=='bca']
        fig.add_scatter(x=[f's{row["master_seed"]}/i{row["outer_id"]}' for row in rows],
            y=[float(row['e']) for row in rows], mode='markers', name=LABELS[distribution],
            marker=dict(color=COLORS[distribution],size=8))
    fig.add_hline(y=.10,line_dash='dash',annotation_text='0.10SE')
    fig.update_layout(template='plotly_dark',title='BCa：全60原標本の保存e（上端点／下端点RMSEの最大）',yaxis_title='標本SE単位',height=420)
    figures.append(fig)
    for example in data['representatives']['examples']:
        selected = load_verified(evidence, root, example['distribution'], int(example['master_seed']), int(example['outer_id']))
        for endpoint in ENDPOINTS:
            fig = endpoint_figure(selected, 'bca', endpoint)
            fig.update_layout(title=f'{LABELS[example["distribution"]]} · {example["selection"]} e例 · seed{example["master_seed"]}/ID{example["outer_id"]} · {endpoint}')
            figures.append(fig)
    groups = data['hypotheses']['groups']
    header = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>MATHSCAPE RQ-M11</title><body style="background:#10151f;color:#eee;font:16px sans-serif;max-width:1100px;margin:30px auto"><h1>全列挙参照とのずれ · RQ-M11</h1>'
    both_below = all(group['status'] == 'not_supported' for group in groups.values())
    verdict = '両群とも27/30基準に未達' if both_below else STATUS[data['hypotheses']['status']]
    header += f'<p>BCa：G {groups["gamma"]["pass_count"]}/30、L {groups["lognormal"]["pass_count"]}/30 が0.10SE以内。{verdict}。固定保存データの記述的判断です。</p>'
    header += '<p>n5・B4999・各原標本R20×120系列。保存binary64の3125順序付き添字による条件付き有限分布を参照します。全60個のeと、固定した各分布最小／最大BCa例の上下2400端点分布だけを表示します。全144000系列の図示ではありません。白実線=逆CDF参照、紫破線=linear参照。母集団真値・90%保証・M10保留の原因説明ではありません。</p>'
    html = header + ''.join(pio.to_html(fig, full_html=False, include_plotlyjs=True if i==0 else False,
        config={'displaylogo':False}, div_id=f'rq-m11-{i}') for i,fig in enumerate(figures)) + '</body></html>'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(html)
    return path
