"""RQ-M12 presentation of audited saved arrays; no numerical science imports."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/rq_m12'
ENDPOINTS = ('lower', 'upper')
LABELS = {'gamma': 'G：標準化ガンマ', 'lognormal': 'L：標準化対数正規'}
STATUS = {'supported': '記述的支持', 'not_supported': '記述的不支持', 'inconclusive': '保留'}
COLORS = {'gamma': '#FFB86B', 'lognormal': '#46DCC0'}
TABLES = ('original_metrics', 'endpoint_metrics', 'summary', 'classification', 'classification_ids', 'reasons')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_bytes())


def _npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _rows(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding='utf-8'))))


def _safe_relative(path):
    path = Path(path)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe saved receipt path')
    return path


def load_verified(evidence=EVIDENCE, root=ROOT, distribution='gamma', seed=0, outer=0):
    """Bind every consumed saved product to the completed scientific audit."""
    evidence, root = Path(evidence), Path(root)
    if distribution not in LABELS or seed not in (0, 1, 2) or outer not in range(10):
        raise ValueError('saved identity outside fixed design')
    audit = _json(evidence/'independent_audit.json')
    main, checked = (_json(evidence/name) for name in ('main_resources.json', 'audit_resources.json'))
    receipt, manifest = (_json(evidence/name) for name in ('input_receipt.json', 'manifest.json'))
    if (audit.get('question_id') != 'RQ-M12' or audit.get('status') != 'passed' or audit.get('mismatch_count') != 0 or
        any(item.get('status') != 'completed' or item.get('exit_code') != 0 for item in (main, checked))):
        raise ValueError('completed main and matching independent audit are required')
    commit = receipt['preanalysis_commit']
    if any(item.get('preanalysis_commit') != commit for item in (main, checked, audit)):
        raise ValueError('analysis and audit source receipts differ')
    if (receipt['question_id'] != 'RQ-M12' or manifest['question_id'] != 'RQ-M12' or
        receipt['accepted_head'] != manifest['accepted_head'] or
        receipt['manifest_sha256'] != _digest(evidence/'manifest.json') or
        receipt['inputs'] != manifest['source_inputs'] or receipt['originals'] != manifest['originals']):
        raise ValueError('frozen input receipt differs')
    if set(audit['source_sha256']) != set(receipt['inputs']):
        raise ValueError('audited source inventory differs')
    for relative, recorded in receipt['inputs'].items():
        path = root/_safe_relative(relative)
        if audit['source_sha256'][relative] != recorded['sha256'] or _digest(path) != recorded['sha256']:
            raise ValueError('audited accepted source bytes differ: '+relative)
    stem = f'{distribution}_seed{seed}_outer{outer}.npz'
    products = [name+'.csv' for name in TABLES]
    products += ['decision.json', 'representatives.json', 'analysis_resources.json',
                 'input_receipt.json', 'manifest.json', 'series/'+stem]
    for relative in products:
        if audit['output_sha256'].get(relative) != _digest(evidence/_safe_relative(relative)):
            raise ValueError('audited saved output differs: '+relative)
    data = _npz(evidence/'series'/stem)
    panel = _npz(root/'evidence/rq_m10/panels'/f'{distribution}_seed{seed}.npz')
    sample = panel['samples'][outer]
    digest = hashlib.sha256(np.asarray(sample, dtype='<f8').tobytes()).hexdigest()
    if (str(data['question_id']) != 'RQ-M12' or str(data['source_question_id']) != 'RQ-M11' or
        str(data['distribution']) != distribution or int(data['master_seed']) != seed or
        int(data['outer_id']) != outer or str(data['sample_sha256']) != digest or
        panel['sample_sha256'][outer].decode() != digest or int(data['n']) != 5 or
        int(data['B']) != 4999 or int(data['R']) != 20 or int(data['K']) != 120 or
        tuple(data['endpoints'].astype(str)) != ENDPOINTS):
        raise ValueError('selected identity or saved design differs')
    if (data['g'].shape != (2400,2) or data['valid'].shape != (2400,2) or
        data['valid'].dtype != bool or data['selected_index'].dtype != np.int32 or
        data['cdf_probability'].shape != data['support'].shape):
        raise ValueError('selected saved array axes differ')
    result = {name: _rows(evidence/(name+'.csv')) for name in TABLES}
    decision = _json(evidence/'decision.json')
    if decision['question_id'] != 'RQ-M12' or decision['groups'] != audit['groups'] or decision['computed_status'] != audit['computed_status']:
        raise ValueError('audited saved decision differs')
    result.update(data=data, decision=decision, representatives=_json(evidence/'representatives.json'),
        audit=audit, evidence=evidence, distribution=distribution, seed=seed, outer=outer,
        digest=digest, sample=sample)
    return result


def _cell(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return repr(value) if np.isfinite(value) else 'NA'
    return value


def series_csv(bundle):
    """Lossless formatting of persisted series; no inverse-CDF or metrics."""
    data = bundle['data']
    metadata = dict(distribution=bundle['distribution'], master_seed=bundle['seed'], outer_id=bundle['outer'],
        sample_sha256=bundle['digest'], n=5, alpha=.05, B=4999, R=20, K=120,
        reference_convention='fixed M11 binary64 conditional inverse CDF; exact integer selection')
    rows = []
    for index in range(2400):
        row = dict(metadata, series_id=index)
        for key, values in data.items():
            if key in ('support','frequency','cumulative','cdf_probability','reason_labels','endpoints'):
                continue
            if values.shape == (2400,):
                row[key] = values[index]
            elif values.shape == (2400,2):
                for ei, endpoint in enumerate(ENDPOINTS):
                    row[endpoint+'_'+key] = values[index,ei]
            elif values.shape == (2,):
                for ei, endpoint in enumerate(ENDPOINTS):
                    row[endpoint+'_'+key] = values[ei]
            elif values.shape == ():
                row[key] = values.item()
        for ei, endpoint in enumerate(ENDPOINTS):
            row[endpoint+'_reason'] = data['reason_labels'][data['reason_code'][index,ei]]
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fields)
    writer.writeheader()
    writer.writerows({key:_cell(value) for key,value in row.items()} for row in rows)
    return buffer.getvalue().encode('utf-8')


def exports(bundle):
    evidence = bundle['evidence']
    prefix = f'rq_m12_{bundle["distribution"]}_seed{bundle["seed"]}_outer{bundle["outer"]}'
    return (
        ('M12全60原標本CSVを保存', (evidence/'original_metrics.csv').read_bytes(), 'rq_m12_original_metrics.csv', 'text/csv'),
        ('M12全120端点指標CSVを保存', (evidence/'endpoint_metrics.csv').read_bytes(), 'rq_m12_endpoint_metrics.csv', 'text/csv'),
        ('M12群別要約CSVを保存', (evidence/'summary.csv').read_bytes(), 'rq_m12_summary.csv', 'text/csv'),
        ('選択標本の2400系列・両端点CSVを保存', series_csv(bundle), prefix+'_series.csv', 'text/csv'),
        ('M12群判定・4分類JSONを保存', (evidence/'decision.json').read_bytes(), 'rq_m12_decision.json', 'application/json'))


def _selected(bundle, table):
    return [row for row in bundle[table] if row['distribution']==bundle['distribution'] and
            int(row['master_seed'])==bundle['seed'] and int(row['outer_id'])==bundle['outer']]


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '未定義'
    return format(number,'.17g') if np.isfinite(number) else '未定義'


def cdf_figure(bundle, endpoint, series_id):
    """Plot only stored probabilities, support points, and selected A/C/Y."""
    data, ei = bundle['data'], ENDPOINTS.index(endpoint)
    fig = go.Figure(go.Scatter(x=data['support'], y=data['cdf_probability'], mode='lines',
        line=dict(shape='hv',color=COLORS[bundle['distribution']]), name='固定した全列挙CDF'))
    for name, x, y, color, symbol in (
        ('A = Q(q*)', data['A'][ei], data['q_star'][ei], '#A8A2FF', 'diamond'),
        ('C = Q(q_B)', data['C'][series_id,ei], data['q_B'][series_id,ei], '#46DCC0', 'circle'),
        ('Y：保存有限B端点', data['Y'][series_id,ei], data['q_B'][series_id,ei], '#FFB86B', 'x')):
        if np.isfinite(x) and np.isfinite(y):
            fig.add_scatter(x=[float(x)],y=[float(y)],mode='markers',name=name,
                marker=dict(size=12,color=color,symbol=symbol),hovertemplate='%{x:.17g}<br>q=%{y:.17g}<extra>'+name+'</extra>')
    fig.update_layout(template='plotly_dark',height=420,title='同じ段差上でq*をq_Bに置き換える',
        xaxis_title='条件付き平均の支持点／計算端点（観測単位）',yaxis_title='累積確率／選択確率',
        legend=dict(orientation='h',y=1.16),margin=dict(t=90,b=45,l=50,r=20))
    return fig


def _remember_selection(state, distribution):
    state['movement_selections'][distribution].update(endpoint=st.session_state.movement_endpoint,
                                                     stream=st.session_state.movement_stream)


def _choose_example(state, example):
    state['distribution'] = example['distribution']
    selection = state['selections'][example['distribution']]
    selection.update(seed=int(example['master_seed']),outer=int(example['outer_id']))
    for key,value in dict(distribution=example['distribution'],seed=selection['seed'],outer=selection['outer'],r=selection['r']).items():
        st.session_state['au_'+key]=value


def render_movement_view(state):
    st.subheader('確率の揺れは、端点をどれだけ動かす？')
    selection = state['selections'][state['distribution']]
    try:
        bundle = load_verified(distribution=state['distribution'],seed=selection['seed'],outer=selection['outer'])
    except (OSError,KeyError,ValueError,TypeError,IndexError) as error:
        st.error(f'M12は未検証・保留です。保存解析／監査との対応を確認できません（{type(error).__name__}）。')
        return
    groups = bundle['decision']['groups']
    for column, distribution in zip(st.columns(2), LABELS):
        row = groups[distribution]
        column.metric(LABELS[distribution]+' · M12 0.10SE以内',f'{row["pass_count"]} / {row["planned_N"]}')
        column.caption(f'完備{row["complete_N"]} ／ 必要無効{row["invalid_N"]} ／ 欠測{row["missing_N"]} ／ 判定不明{row["unknown_N"]}。計画分母は{row["planned_N"]}を保持。')
    st.info('H12-Q：'+STATUS[bundle['decision']['status']]+'。必要30標本がすべて完備し、上下RMSがともに0.10標本SE以内の標本が27/30以上という記述基準です。')
    st.caption('M11の結果を見た後の保存データ再解析です。固定CDF上でq*だけをq_Bに替えた移動gを扱い、総ずれや被覆率、M10保留の原因とは区別します。BCaのみが主対象で、percentileのg=0は実装対照です。')
    st.success(f'{LABELS[bundle["distribution"]]} ／ seed{bundle["seed"]} ／ 原標本ID{bundle["outer"]} ／ n5・α=.05・B4999・R20×120共通系列')
    st.caption('原標本SHA-256：'+bundle['digest']+'。3配分で系列を重複計上しません。旧配分・旧手法の選択は復帰用に保持します。')
    state.setdefault('movement_selections',{name:dict(endpoint='lower',stream=0) for name in LABELS})
    stored = state['movement_selections'][state['distribution']]
    if st.session_state.get('movement_distribution') != state['distribution']:
        st.session_state.movement_endpoint=stored['endpoint']
        st.session_state.movement_stream=stored['stream']
        st.session_state.movement_distribution=state['distribution']
    st.session_state.setdefault('movement_endpoint',stored['endpoint'])
    st.session_state.setdefault('movement_stream',stored['stream'])
    endpoint=st.selectbox('移動を表示する端点',ENDPOINTS,key='movement_endpoint',
        on_change=_remember_selection,args=(state,state['distribution']))
    stream=st.number_input('移動を表示する系列 k',min_value=0,max_value=119,step=1,key='movement_stream',
        on_change=_remember_selection,args=(state,state['distribution']))
    # r*120+k indexes the persisted r-major array; no scientific quantity is recomputed.
    index=int(selection['r'])*120+int(stream)
    data,ei=bundle['data'],ENDPOINTS.index(endpoint)
    original=_selected(bundle,'original_metrics')[0]
    endpoints=_selected(bundle,'endpoint_metrics')
    for column,label,value in zip(st.columns(3),('選択原標本の u（SE単位）','下端の移動RMS（SE単位）','上端の移動RMS（SE単位）'),
                                  (original['u'],original['lower_r'],original['upper_r'])):
        column.metric(label,_number(value))
    st.caption('保存合否='+original['pass_flag']+' ／ 余裕0.10−u='+_number(original['margin'])+'。判定は丸め前の保存値で行い、画面の丸めや許容差を使いません。')
    st.plotly_chart(cdf_figure(bundle,endpoint,index),width='stretch',config={'displaylogo':False})
    code=int(data['movement_code'][index,ei])
    label={-2:'未定義',-1:'下へ移動',0:'同じ支持点',1:'上へ移動'}[code]
    qlabel='qは変化' if bool(data['q_changed'][index,ei]) else 'qは不変' if bool(data['valid'][index,ei]) else 'qの判定は未定義'
    st.write(f'選択系列 r={selection["r"]}, k={stream}（保存ID{index}） · {endpoint}：{qlabel} ／ {label}。g='+_number(data['g'][index,ei])+' 標本SE')
    selected_rows=[]
    for end_index,end in enumerate(ENDPOINTS):
        row=dict(endpoint=end)
        for key in ('q_star','A','A_index'):
            row[key]=_cell(data[key][end_index])
        for key in ('q_B','q_delta','C','Y','selected_index','support_step','g','h','d','valid','movement_code','q_changed','q_changed_same','endpoint_identity_residual','source_tail_count'):
            row[key]=_cell(data[key][index,end_index])
        row['source_warning']=bool(data['source_warning'][index])
        row['reason']=str(data['reason_labels'][data['reason_code'][index,end_index]])
        selected_rows.append(row)
    st.dataframe(selected_rows,hide_index=True,width='stretch')
    st.caption('同一点／上下の分類は保存支持点の厳密な同一性です。qが違っても同じ段に留まれます。近接した別の支持点への移動を0へ丸めず、上表・CSVで17桁精度の値を確認できます。Yは有限Bの別の経験分布から得たlinear端点で、このCDF上の点とは限りません。')
    fields=('endpoint','same_count','down_count','up_count','same_rate','moved_abs_g_mean','mean_g','rms_g','abs_g_p95','abs_g_max','q_changed_same_count','source_warning_series','invalid_series','missing_series')
    st.dataframe([{key:row[key] for key in fields} for row in endpoints],hide_index=True,width='stretch')
    with st.expander('固定したu最小／最大の代表例'):
        for example in bundle['representatives']['examples']:
            st.button(f'{LABELS[example["distribution"]]} · {example["selection"]} u · seed{example["master_seed"]}/ID{example["outer_id"]}',
                key='movement_example_'+example['distribution']+example['selection'],on_click=_choose_example,args=(state,example))
    with st.expander('G/L・上下の要約と全60標本、M11との4分類'):
        for name in ('summary','original_metrics','endpoint_metrics','classification','classification_ids'):
            st.dataframe(bundle[name],hide_index=True,width='stretch')
        st.caption('要約は原標本ごとの量の等重み平均です。2400を原標本Nへ加えず、移動0件の移動時平均はNAのまま、定義された原標本数を併記します。全60標本を対象としM11不合格だけに絞りません。')
    with st.expander('g・h・d、交差項、無効理由と監査範囲'):
        st.latex(r'd=g+h,\qquad\overline{d^2}=\overline{g^2}+\overline{h^2}+2\overline{gh}')
        st.write('gは同じ条件付き分布で選択確率を置き換えた移動です。hには有限B経験分布と分位点補間規約の差がともに入り、同じ系列のgとhは依存し得ます。交差項は符号付きで、因果的寄与率や独立な成分とは扱いません。')
        st.dataframe(endpoints,hide_index=True,width='stretch')
        st.dataframe(bundle['reasons'],hide_index=True,width='stretch')
        st.caption('参照は保存binary64平均規約による条件付き有限分布で、母集団真値ではありません。有効なq=0/1・低tail警告は含め、必要無効や欠測は計画N30を保って保留とします。M08/M09のn20支持、M10保留、M11記述的不支持を更新しません。')
        st.caption('独立監査は全288000端点と保存集計・分類を照合し不一致0。生成経路まで再構成したのは既存36添字系列のみで、残り143964系列は保存入力以降の確認です。新標本・新bootstrap・PRNG再生・M11主解析再実行は0です。')
    for label,payload,filename,mime in exports(bundle):
        st.download_button(label,payload,file_name=filename,mime=mime,key='movement_download_'+filename)
