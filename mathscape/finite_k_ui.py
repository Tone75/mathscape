"""Read-only finite-K measurement view; all numerical results are saved evidence."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence/rq_m07'
M06=ROOT/'evidence/rq_m06'
M05=ROOT/'evidence/rq_m05'
STATE_KEY='finite_k_saved_view'
LABELS={'gamma':'G：標準化ガンマ','lognormal':'L：標準化対数正規'}
METHODS={'t':'t','percentile':'percentile','bca':'BCa'}
COLORS={'t':'#46DCC0','percentile':'#FFB86B','bca':'#A8A2FF'}


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY]=dict(distribution='gamma',K=120,method='bca',selections={dist:dict(seed=0,outer=0,r=0) for dist in LABELS})
    return st.session_state[STATE_KEY]


def _population():
    state=_state();state['distribution']=st.session_state.fk_distribution
    for key,value in state['selections'][state['distribution']].items():st.session_state[f'fk_{key}']=value


def _remember():
    state=_state()
    for key in ('seed','outer','r'):
        if f'fk_{key}' in st.session_state:state['selections'][state['distribution']][key]=st.session_state[f'fk_{key}']
    for key in ('K','method'):
        if f'fk_{key}' in st.session_state:state[key]=st.session_state[f'fk_{key}']


@st.cache_data(max_entries=4,show_spinner=False)
def _panel(path,modified):
    with np.load(path,allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}


@st.cache_data(max_entries=12,show_spinner=False)
def _rows(path,modified):
    path=Path(path);payload=path.read_bytes();text=gzip.decompress(payload) if path.suffix=='.gz' else payload
    return list(csv.DictReader(io.StringIO(text.decode('utf-8')))),payload


@st.cache_data(max_entries=6,show_spinner=False)
def _hashes(path,modified):
    with np.load(path,allow_pickle=False) as saved:return saved['sample_sha256'].astype(str)


def _load(name):
    path=EVIDENCE/name
    return _rows(str(path),path.stat().st_mtime_ns)


def _number(value,fmt='.6g'):
    return format(float(value),fmt) if value is not None and np.isfinite(float(value)) else '未定義'


def _display(rows):
    output=[];mixed_seed=any(row.get('master_seed')=='pooled' for row in rows)
    for row in rows:
        converted={}
        for key,value in row.items():
            if key=='master_seed' and mixed_seed:converted[key]=str(value);continue
            if value=='':converted[key]=None;continue
            if value in ('True','False'):converted[key]=value=='True';continue
            try:
                number=float(value);converted[key]=number if np.isfinite(number) else None
            except (TypeError,ValueError):converted[key]=value
        output.append(converted)
    return output


def _chart(fig,x,y,height=340):
    fig.update_layout(template='plotly_dark',height=height,xaxis_title=x,yaxis_title=y,margin=dict(l=40,r=25,t=65,b=45),
                      paper_bgcolor='rgba(0,0,0,0)',legend=dict(orientation='h',y=1.18))
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False})


def _intervals(panel,state,selection):
    oi,rep=selection['outer'],selection['r'];mi=list(METHODS).index(state['method']);sample=panel['samples'][oi]
    fig=go.Figure(go.Scatter(x=sample,y=np.full(20,-7),mode='markers',name='固定した保存原標本',marker=dict(symbol='circle',size=8,color='white',line=dict(color='#080C14',width=1))))
    if panel['valid'][oi,rep,0,0]:
        fig.add_scatter(x=[panel['lower'][oi,rep,0,0],panel['upper'][oi,rep,0,0]],y=[-3,-3],mode='lines+markers',name='t参照（実質1区間）',line=dict(color=COLORS['t'],width=3),marker=dict(size=7))
    for start,stop,label,dash in ((0,30,'先頭30','solid'),(30,120,'追加90','dot')):
        if start>=state['K'] or state['method']=='t':continue
        x,y=[],[]
        for inner in range(start,stop):
            if panel['valid'][oi,rep,mi,inner]:
                x.extend([panel['lower'][oi,rep,mi,inner],panel['upper'][oi,rep,mi,inner],None]);y.extend([inner,inner,None])
        fig.add_scatter(x=x,y=y,mode='lines',name=f'{METHODS[state["method"]]} · {label}',line=dict(color=COLORS[state['method']],width=2,dash=dash))
    allx=np.concatenate([sample,panel['lower'][oi].ravel(),panel['upper'][oi].ravel()]);allx=allx[np.isfinite(allx)]
    low,high=float(allx.min()),float(allx.max());padding=max((high-low)*.05,1e-6)
    fig.update_xaxes(range=[low-padding,high+padding],autorange=False);fig.add_vline(x=0,line_dash='dot',line_color='white',annotation_text='真の平均 μ=0')
    _chart(fig,'保存95%推論区間の端点／観測値（原標本固定の共通軸）','系列ID k（下部：原標本／t参照）',560)
    mask=panel['valid'][oi,rep,mi,:state['K']];warnings=panel['tail_warning'][oi,rep,mi,:state['K']]
    st.caption(f'表示r={rep}・{METHODS[state["method"]]}・K={state["K"]}：論理行{len(mask)}、無効{int((~mask).sum())}、tau<10警告{int(warnings.sum())}。'
               'tのbroadcastは独立系列ではなく実質1区間です。横軸は同じ原標本の全r・全K・全手法・観測値から固定し、K・r・手法で変更しません。')
    if not mask.all():
        reasons=panel['reason_labels'][panel['reason_codes'][oi,rep,mi,:state['K']][~mask]].astype(str)
        st.warning('無効な保存区間を描画から除外：'+' / '.join(sorted(set(reasons)))+'。指標のNAと主分母は保持し、他手法への置換はありません。')
    st.info('95%推論区間は原標本から平均を推論する区間です。Aseを測り直したときのばらつきや、後段の原標本単位99%比較区間とは別の量です。')


def _measurement(originals,reps,krows,groups,state,selection):
    method=state['method'];current=next(row for row in krows if row['method']==method and int(row['replicate_id'])==selection['r'] and int(row['K'])==state['K'])
    st.caption('選択原標本・選択r・選択方法・Kの保存済み測定値。seは同じ原標本から固定。')
    for labels,keys in ((('下端点SD','上端点SD','合成変動 S'),('lower_sd','upper_sd','S')),(('標本SE（s/√20）','測定Ase','測定Apop'),('se','Ase','Apop'))):
        for column,label,key in zip(st.columns(3),labels,keys):column.metric(label,_number(current[key]))
    selected=[row for row in reps if row['method']==method];fig=go.Figure()
    for kval,color in ((30,'#FFB86B'),(120,'#A8A2FF')):
        fig.add_scatter(x=[int(row['replicate_id']) for row in selected],y=[float(row[f'A{kval}']) for row in selected],mode='lines+markers',name=f'Ase K{kval}'+('（先頭組）' if kval==30 else '（有限）'),line=dict(color=color,width=4 if state['K']==kval else 1),marker=dict(size=6))
    _chart(fig,'測定反復 r（同じ原標本・独立系列）','各反復のAse（20組の対応値）')
    original=next(row for row in originals if row['method']==method)
    fields=['method','complete','planned_replicates','complete_replicates','d','Q30','Q120','W','ratio_Q120_Q30','mean_A30','mean_A120','total_inner','valid_inner','invalid_inner']
    st.dataframe(_display([{key:row[key] for key in fields} for row in originals]),hide_index=True,width='stretch')
    st.caption('Q30/Q120は20回測ったAseの反復分散（ddof=1）。Vは1回の測定内の端点分散を標本SE²で規格化した量で、Qや原標本間MCSEとは異なります。'
               't対照のA/V/Qはゼロ、Q30=0の比はNAです。主K30は常に先頭30系列で、4組平均に置き換えません。')
    group_rows=[row for row in groups if row['method']==method and int(row['replicate_id'])==selection['r']]
    st.dataframe(_display([{key:row[key] for key in ('group','group_start','group_stop','lower_mean','upper_mean','lower_var','upper_var','V','A','complete')} for row in group_rows]),hide_index=True,width='stretch')
    rep=next(row for row in selected if int(row['replicate_id'])==selection['r']);gap=go.Figure()
    gap.add_bar(x=['Jgap','Pgap','Rgap'],y=[float(rep[key]) for key in ('Jgap','Pgap','Rgap')],name='符号を保った3項')
    gap.add_scatter(x=['delta'],y=[float(rep['delta'])],mode='markers',name='A120−A30先頭',marker=dict(size=12,color='white'))
    gap.add_hline(y=0,line_dash='dot');_chart(gap,'選択rの代数的分解','Ase単位の差',300)
    fields=['V30','A30','V120','A120','Vbar','Abar','Jgap','Pgap','Rgap','delta','decomposition_residual','lower_ss_residual','upper_ss_residual','decomposition_normalized_residual','lower_ss_normalized_residual','upper_ss_normalized_residual','algebra_ok','reason']
    st.dataframe(_display([{key:rep[key] for key in fields}]),hide_index=True,width='stretch')
    st.latex(r'\Delta=A_{120}-A_{30,\mathrm{first}}=J_{\mathrm{gap}}+P_{\mathrm{gap}}+R_{\mathrm{gap}}')
    st.caption('4組はk0–29／30–59／60–89／90–119。Vbar=4組Vの平均、Abar=4組√Vの平均。Jgap=√Vbar−Abar、Pgap=√V120−√Vbar、Rgap=Abar−A30先頭。'
               'Jgap≥0は平方根の凹性による代数的性質で、科学仮説の支持ではありません。V120は4組Vの単純平均とは異なり、各端点の平方和分解も保存・監査しています。'
               'K120も有限で真値ではなく、Jgapを無限Kに対する絶対偏りとは呼べません。')


def _pooled(rows,seed_rows,decomposition,state):
    method=state['method'];fig=go.Figure()
    for dist,color in (('gamma','#46DCC0'),('lognormal','#FFB86B')):
        selected=[next(row for row in rows if row['distribution']==dist and row['method']==method and int(row['K'])==kval) for kval in (30,120)]
        values=np.array([float(row['A']) for row in selected]);high=np.array([float(row['A_decision_high']) for row in selected]);low=np.array([float(row['A_decision_low']) for row in selected])
        bars=dict(type='data',symmetric=False,array=high-values,arrayminus=values-low) if np.isfinite(np.r_[values,high,low]).all() else None
        fig.add_scatter(x=['30先頭','120有限'],y=values,mode='lines+markers',name=LABELS[dist],line_color=color,error_y=bars)
    _chart(fig,'系列数 K（選択方法）','平均Aseと原標本単位99%近似区間')
    st.caption('tの99%比較区間はSE0でNA、点推定0のみを表示します。その他も非有限の比較区間は描かず、表のNAと分母を保持します。')
    compact=['distribution','method','K','original_total','A','A_mcse','A_p05','A_p50','A_p95','V','Q','C','T','mean_width','warning_rate','interval_count','invalid_count','missing_original_count']
    st.dataframe(_display([{key:row[key] for key in compact} for row in rows]),hide_index=True,width='stretch')
    st.info('各分布の独立推論単位N=30、各seedはN=10。合計60原標本はM06／M05保存原標本の再利用で、独立な新規追試ではありません。'
            'J20・K120・4組をNに足しません。99%区間は保存IID原標本を固定IDで選ぶモデルに基づく原標本間近似区間で、有限J/Kのノイズを含みます。'
            '固定60標本を条件とする内側乱数だけの厳密区間ではありません。N30・原標本別量の非正規性により近似が不十分な可能性があります。'
            'C/T/幅/警告は記述値です。M05/M06の判断を再検定・更新・プールしません。')
    with st.expander('保存済みプール全12行（全分位点・conditional_・分母）'):st.dataframe(_display(rows),hide_index=True,width='stretch')
    with st.expander('保存済みseed別全36行（各N10・記述）'):st.dataframe(_display(seed_rows),hide_index=True,width='stretch')
    with st.expander('全符号・残差・比の定義（保存済み分解集計24行）'):
        st.dataframe(_display(decomposition),hide_index=True,width='stretch')
        st.caption('mean_individual_Q120_Q30は原標本ごとの比の平均、ratio_mean_Q120_over_mean_Q30は平均Qの比。異なる量で、0分母はNA。'
                   'conditional_は定義できた完全例だけの参考で主判定に使いません。raw符号・分位点は記述であり、rを独立原標本とは数えません。')


def _hypotheses(saved):
    st.subheader('4. 事前固定した4本の主比較（BCa・各分布N30）')
    for H,field,axis in (('H1','d','原標本平均d：A120−A30先頭（Ase単位）'),('H2','W','原標本平均W：Q120−0.5Q30（Ase²・反復分散）')):
        fig=go.Figure();records=[]
        for dist in LABELS:
            row=saved['quantities'][f'{H}_{dist}']
            if row['valid']:
                fig.add_scatter(x=[row['mean']],y=[LABELS[dist]],name=LABELS[dist],mode='markers',error_x=dict(type='data',symmetric=False,array=[row['high']-row['mean']],arrayminus=[row['mean']-row['low']]))
            records.append(dict(母集団=LABELS[dist],N=row['N'],finite_count=row['finite_count'],mean=row['mean'],MCSE=row['mcse'],lower99=row['low'],upper99=row['high'],判定=row['status']))
        if H=='H1':fig.add_vrect(x0=-.0025,x1=.0025,fillcolor='#46DCC0',opacity=.15,line_width=0,annotation_text='±.0025 Ase単位')
        fig.add_vline(x=0,line_dash='dot',line_color='white');_chart(fig,axis,H+'：99%原標本t区間',280)
        st.dataframe(records,hide_index=True,width='stretch')
        status=saved['hypotheses'][H]['status'];text={'supported':'支持（両分布が支持）','not_supported':'不支持（少なくとも一方が不支持）','inconclusive':'保留'}[status]
        st.write(f'{H}全体：{text}。')
    st.caption('H1は各99%区間が厳密に±.0025内で支持、帯の片側へ完全に外れれば不支持、接触・交差は保留。'
               '.0025は設計上のAse単位の許容帯で、被覆率0.25ppではありません。H2は上限<0で支持、下限≥0で不支持、その他は保留。'
               '50%はAseの反復分散Qの低減で、SDの50%低減ではありません。原標本ごとの比の平均を主判定に使いません。'
               '必要欠測・無効・主比較SE0はNA/保留。各仮説は両分布支持だけ全体支持、いずれか不支持があれば全体不支持。seed符号ゲートはありません。'
               '確認的区間は4本の近似99%区間のみ。和事象上界の非被覆率約4%以下という意図は近似の妥当性を前提とし、有限標本の厳密保証ではありません。')


def _sample_csv(sample,state,selection,digest):
    buffer=io.StringIO();writer=csv.writer(buffer);writer.writerow(['distribution','master_seed','outer_id','observation_index','X','sample_sha256','source_question'])
    for index,value in enumerate(sample):writer.writerow([state['distribution'],selection['seed'],selection['outer'],index,float(value),digest,'RQ-M06 / RQ-M05'])
    return buffer.getvalue().encode()


def _endpoint_csv(panel,state,selection,digest):
    buffer=io.StringIO();fields=['distribution','master_seed','outer_id','replicate_id','inner_id','method','B','sample_sha256','lower','upper','valid','reason','a','z0','q_low','q_high','tau_low','tau_high','tail_warning']
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();oi=selection['outer']
    for rep in range(20):
        for mi,method in enumerate(METHODS):
            for inner in range(120):
                row=dict(distribution=state['distribution'],master_seed=selection['seed'],outer_id=oi,replicate_id=rep,inner_id=inner,method=method,B=4999,sample_sha256=digest,
                         reason=str(panel['reason_labels'][panel['reason_codes'][oi,rep,mi,inner]]))
                row.update({key:panel[key][oi,rep,mi,inner].item() for key in fields[8:] if key!='reason'});writer.writerow(row)
    return buffer.getvalue().encode()


def render_finite_k_view():
    st.header('Inference Lab · 揺れの測り方にも揺れがある')
    st.caption('保存結果専用 · n20・固定α=.05・B4999・J20・K30先頭／K120。保存60原標本を使い、新しい原標本やbootstrap系列は生成しません。')
    state=_state();selection=state['selections'][state['distribution']]
    for key,value in dict(distribution=state['distribution'],K=state['K'],method=state['method'],**selection).items():st.session_state.setdefault(f'fk_{key}',value)
    st.sidebar.selectbox('測定精度の母集団',list(LABELS),format_func=LABELS.get,key='fk_distribution',on_change=_population)
    st.sidebar.selectbox('測定精度のmaster seed',[0,1,2],key='fk_seed',on_change=_remember)
    st.sidebar.number_input('測定精度の原標本ID',0,9,step=1,key='fk_outer',on_change=_remember)
    st.sidebar.selectbox('測定反復 r',list(range(20)),key='fk_r',on_change=_remember)
    st.sidebar.selectbox('比較する系列数 K',[30,120],key='fk_K',on_change=_remember)
    st.sidebar.selectbox('測定精度の手法',list(METHODS),format_func=METHODS.get,key='fk_method',on_change=_remember)
    st.sidebar.caption('B4999・n20・α=.05・J20固定。分布別seed／ID／rを復元し、K・方法・Lab・モードの切替でも同じ原標本を保持します。分布変更は独立した別原標本で、同じIDでも対応群ではありません。')
    selection=state['selections'][state['distribution']]
    try:
        path=EVIDENCE/'panels'/f'{state["distribution"]}_seed{selection["seed"]}.npz';panel=_panel(str(path),path.stat().st_mtime_ns)
        tables={name:_load(name)[0] for name in ('original_metrics.csv','original_K_metrics.csv','replicate_metrics.csv.gz','group_metrics.csv.gz','K_metrics.csv.gz','summary.csv','decomposition_summary.csv')}
        pooled,pooled_bytes=_load('pooled_summary.csv');hypothesis_bytes=(EVIDENCE/'hypotheses.json').read_bytes();hypotheses=json.loads(hypothesis_bytes)
        sample=panel['samples'][selection['outer']];digest=hashlib.sha256(np.asarray(sample,dtype='<f8').tobytes()).hexdigest()
        hashes=[]
        for source in (M06,M05):
            old=source/'panels'/path.name;hashes.append(_hashes(str(old),old.stat().st_mtime_ns)[selection['outer']])
        if not digest==panel['sample_sha256'][selection['outer']].decode('ascii')==hashes[0]==hashes[1]:raise ValueError('M06/M05 original provenance hash mismatch')
        old_rows,_=_rows(str(M05/'pooled_summary.csv'),(M05/'pooled_summary.csv').stat().st_mtime_ns)
        if panel['samples'].shape!=(10,20) or panel['lower'].shape!=(10,20,3,120) or not panel['observed_streams'].all():raise ValueError('saved fixed panel shape/completion differs')
    except (OSError,KeyError,ValueError,IndexError) as error:
        st.error(f'RQ-M07保存結果／M06・M05入力対応を表示できません：{type(error).__name__}。証拠ファイルを確認してください。');return
    st.success(f'表示条件：{LABELS[state["distribution"]]} ／ seed={selection["seed"]} ／ 原標本ID={selection["outer"]} ／ r={selection["r"]} ／ K={state["K"]} ／ {METHODS[state["method"]]} ／ n20・固定α=.05・B4999・J20')
    st.caption(f'原標本SHA-256：{digest}。M06・M05保存hashと同じouter IDで一致。出典：evidence/rq_m06/panels/{path.name}（M05由来）。受入入力HEAD：{panel["input_accepted_head"].item()}')
    select=lambda rows:[row for row in rows if row['distribution']==state['distribution'] and int(row['master_seed'])==selection['seed'] and int(row['outer_id'])==selection['outer']]
    st.subheader('1. 同じ原標本・同じrの先頭30／全120端点');_intervals(panel,state,selection)
    st.subheader('2. 測定を20回繰り返すとAseはどう動くか')
    _measurement(select(tables['original_metrics.csv']),select(tables['replicate_metrics.csv.gz']),select(tables['K_metrics.csv.gz']),select(tables['group_metrics.csv.gz']),state,selection)
    st.subheader('3. 各分布N30の全条件集計');_pooled(pooled,tables['summary.csv'],tables['decomposition_summary.csv'],state);_hypotheses(hypotheses)
    with st.expander('選択rの端点・z0/a/q/tau診断（警告を含む保存値）'):
        oi,rep=selection['outer'],selection['r'];mi=list(METHODS).index(state['method']);diagnostics=[]
        for inner in range(1 if state['method']=='t' else state['K']):
            item=dict(method=state['method'],replicate_id=rep,inner_id=inner,reason=str(panel['reason_labels'][panel['reason_codes'][oi,rep,mi,inner]]))
            item.update({key:panel[key][oi,rep,mi,inner].item() for key in ('lower','upper','valid','a','z0','q_low','q_high','tau_low','tau_high','tail_warning')});diagnostics.append(item)
        st.dataframe(diagnostics,hide_index=True,width='stretch')
        st.caption('tau=B×min(q,1−q)。tau<10は警告のまま保持し、主比較・集計から除きません。無効理由と未定義値は置換しません。')
    with st.expander('M05の旧K1安定性NAを維持'):
        fields=['distribution','method','K','C','T','lower_sd','upper_sd','S','Ase','Apop']
        st.dataframe(_display([{key:row[key] for key in fields} for row in old_rows if row['distribution']==state['distribution']]),hide_index=True,width='stretch')
        st.caption('M05のK1ではtを含む安定性はNA。M07のtゼロ変動とは異なります。M06の有限K30・B比較の既存結論も保持します。')
    prefix=f'rq_m07_{state["distribution"]}_seed{selection["seed"]}_outer{selection["outer"]}'
    st.download_button('測定精度プールCSV（12行）を保存',pooled_bytes,file_name='rq_m07_pooled_summary.csv',mime='text/csv',key='fk_summary')
    st.download_button('主比較JSON（4区間・全判定）を保存',hypothesis_bytes,file_name='rq_m07_hypotheses.json',mime='application/json',key='fk_hypotheses')
    st.download_button('固定原標本CSV（20観測）を保存',_sample_csv(sample,state,selection,digest),file_name=prefix+'_sample.csv',mime='text/csv',key='fk_sample')
    st.download_button('同じ原標本の全端点CSV（7200行）を保存',_endpoint_csv(panel,state,selection,digest),file_name=prefix+'_endpoints.csv',mime='text/csv',key='fk_endpoints')
    html=EVIDENCE/'figures/overview.html'
    if html.exists():st.download_button('自己完結HTML概要を保存',html.read_bytes(),file_name='rq_m07_overview.html',mime='text/html',key='fk_html')
    st.caption('端点CSVはJ20×3方法×K120の7200論理行。tは同じ参照のbroadcastで実質1区間／原標本です。HTMLはBCa両分布×Kの4条件の平均Ase・Q概要です。'
               '4主区間・全判定は画面／JSON、全12条件はCSVで確認できます。ダウンロード内容は保存値の読み出しで、追加計算や再標本化を行いません。')
