"""Saved-only RQ-M08 shared-budget allocation explorer."""
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
EVIDENCE=ROOT/'evidence/rq_m08'
M07=ROOT/'evidence/rq_m07'
M06=ROOT/'evidence/rq_m06'
M05=ROOT/'evidence/rq_m05'
STATE_KEY='allocation_saved_view'
LABELS={'gamma':'G：標準化ガンマ','lognormal':'L：標準化対数正規'}
METHODS={'t':'t','percentile':'percentile','bca':'BCa'}
ALLOCATIONS={'P30':(4,30),'P60':(2,60),'P120':(1,120)}
COLORS={'P30':'#FFB86B','P60':'#46DCC0','P120':'#A8A2FF'}


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY]=dict(distribution='gamma',allocation='P30',method='bca',selections={dist:dict(seed=0,outer=0,r=0) for dist in LABELS})
    return st.session_state[STATE_KEY]


def _population():
    state=_state();state['distribution']=st.session_state.au_distribution
    for key,value in state['selections'][state['distribution']].items():st.session_state[f'au_{key}']=value


def _remember():
    state=_state()
    for key in ('seed','outer','r'):
        if f'au_{key}' in st.session_state:state['selections'][state['distribution']][key]=st.session_state[f'au_{key}']
    for key in ('allocation','method'):
        if f'au_{key}' in st.session_state:state[key]=st.session_state[f'au_{key}']


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
    oi,rep=selection['outer'],selection['r'];mi=list(METHODS).index(state['method']);sample=panel['samples'][oi];j,k=ALLOCATIONS[state['allocation']]
    fig=go.Figure(go.Scatter(x=sample,y=np.full(20,-7),mode='markers',name='固定した保存原標本',marker=dict(symbol='circle',size=8,color='white',line=dict(color='#080C14',width=1))))
    if panel['valid'][oi,rep,0,0]:
        fig.add_scatter(x=[panel['lower'][oi,rep,0,0],panel['upper'][oi,rep,0,0]],y=[-3,-3],mode='lines+markers',name='t参照（実質1区間）',line=dict(color='#46DCC0',width=3),marker=dict(size=7))
    if state['method']!='t':
        for group in range(j):
            x,y=[],[]
            for stream in range(group*k,(group+1)*k):
                if panel['valid'][oi,rep,mi,stream]:x.extend([panel['lower'][oi,rep,mi,stream],panel['upper'][oi,rep,mi,stream],None]);y.extend([stream,stream,None])
            fig.add_scatter(x=x,y=y,mode='lines',name=f'{state["allocation"]} · 組{group}（k{group*k}–{(group+1)*k-1}）',line=dict(color=COLORS[state['allocation']],width=2,dash=['solid','dot','dash','dashdot'][group]))
    allx=np.concatenate([sample,panel['lower'][oi].ravel(),panel['upper'][oi].ravel()]);allx=allx[np.isfinite(allx)];lo,hi=float(allx.min()),float(allx.max());pad=max((hi-lo)*.05,1e-6)
    fig.update_xaxes(range=[lo-pad,hi+pad],autorange=False);fig.add_vline(x=0,line_dash='dot',line_color='white',annotation_text='真の平均 μ=0')
    _chart(fig,'保存95%推論区間の端点／観測値（原標本固定の共通軸）','共通系列ID k（120系列を各配分で全使用）',550)
    valid=panel['valid'][oi,rep,mi];warnings=panel['tail_warning'][oi,rep,mi]
    st.caption(f'選択r={rep}・{METHODS[state["method"]]}：120論理行、無効{int((~valid).sum())}、tau<10警告{int(warnings.sum())}。'
               '各配分は同じ120系列を重複なく全使用し、別の系列を追加生成しません。横軸は原標本の全r・全手法から固定しています。tのbroadcastは独立120区間ではありません。')
    if not valid.all():
        reasons=panel['reason_labels'][panel['reason_codes'][oi,rep,mi][~valid]].astype(str);st.warning('無効な保存区間：'+' / '.join(sorted(set(reasons)))+'。描画から除外し、主量NAと計画分母を保持します。')
    st.info('95%端点区間は原標本から母平均を推論する区間です。TのR20回の反復ばらつき、後段の原標本単位99.5%比較区間とは区別してください。')


def _measurements(originals,comparisons,budgets,groups,decompositions,state,selection):
    method=state['method'];allocation=state['allocation'];rep=selection['r'];selected=[row for row in budgets if row['method']==method]
    current=next(row for row in selected if row['allocation']==allocation and int(row['budget_repeat_id'])==rep)
    st.caption('選択原標本・選択r・選択配分の保存指標。下端点SD・上端点SD・Sは組別量を平均した値です。')
    for labels,keys in ((('組平均の下端点SD','組平均の上端点SD','組平均のS'),('lower_sd','upper_sd','S')),(('標本SE（s/√20）','予算推定値 T=Ase','平均V=Vbar=Vse'),('se','T','Vbar'))):
        for column,label,key in zip(st.columns(3),labels,keys):column.metric(label,_number(current[key]))
    st.latex(r'T_p=\frac1J\sum_{j=1}^{J}\sqrt{V_{pj}},\qquad Q_p=\operatorname{var}_{r}(T_{rp}),\quad \mathrm{ddof}=1')
    fig=go.Figure()
    for name in ALLOCATIONS:
        rows=[row for row in selected if row['allocation']==name]
        fig.add_scatter(x=[int(row['budget_repeat_id']) for row in rows],y=[float(row['T']) for row in rows],mode='lines+markers',name=name,line=dict(color=COLORS[name],width=4 if name==allocation else 1),marker=dict(size=6))
    _chart(fig,'予算全体の反復 r（R20、対応する共通120系列）','T：組の平方根を取ってから平均したAse')
    compact=['allocation','method','J','K','M','Q','Q_over_R','complete','complete_repeats','planned_repeats','total_streams','valid_streams','invalid_streams']
    st.dataframe(_display([{key:row[key] for key in compact} for row in originals]),hide_index=True,width='stretch')
    st.caption('M=R20回のTの平均、Q=R20回のTの標本分散です。Q/Rは反復平均Mの条件付き分散の参考推定値で、主Qに置き換えません。'
               'Vse=Vbar=mean_j(V)、Ase=T=mean_j(√V)なので、J>1では一般にVse≠Ase²。組SDやSも先に組で計算してから平均し、全端点を先にプールしません。')
    fields=['method','d','L','U','Q30','Q120','M30','M120','ratio_Q120_Q30','complete','complete_repeats','total_streams','valid_streams','invalid_streams']
    st.dataframe(_display([{key:row[key] for key in fields} for row in comparisons]),hide_index=True,width='stretch')
    group_rows=[row for row in groups if row['method']==method and int(row['budget_repeat_id'])==rep]
    groupfig=go.Figure()
    for name in ALLOCATIONS:
        rows=[row for row in group_rows if row['allocation']==name];groupfig.add_bar(x=[f'{name} 組{row["group"]}' for row in rows],y=[float(row['A']) for row in rows],name=name,marker_color=COLORS[name])
    _chart(groupfig,'4×30／2×60／1×120 の固定7組','各組のA=√V（同じ120系列）',300)
    fields=['allocation','J','K','group','group_start','group_stop','lower_var','upper_var','V','A','complete','total_streams','valid_streams','invalid_streams']
    st.dataframe(_display([{key:row[key] for key in fields} for row in group_rows]),hide_index=True,width='stretch')
    decomp=next(row for row in decompositions if row['method']==method and int(row['budget_repeat_id'])==rep);fig=go.Figure()
    fig.add_bar(x=['Jgap','Pgap'],y=[float(decomp[key]) for key in ('Jgap','Pgap')],name='符号を保った2項');fig.add_scatter(x=['delta'],y=[float(decomp['delta'])],mode='markers',name='T_P120−T_P30',marker=dict(size=12,color='white'));fig.add_hline(y=0,line_dash='dot')
    _chart(fig,'選択rの代数的分解（Rgapはありません）','Ase単位の差',280)
    fields=['Vbar30','T30','T60','T120','Jgap','Pgap','delta','residual','normalized_residual','ss30_lower_residual','ss30_upper_residual','ss60_lower_residual','ss60_upper_residual','algebra_ok','reason']
    st.dataframe(_display([{key:decomp[key] for key in fields}]),hide_index=True,width='stretch')
    st.caption('T_P120−T_P30=Jgap+Pgap。Jgap=√mean(V30組)−mean(√V30組)、Pgap=√V120−√mean(V30組)。M07の先頭組との差Rgapは今回存在しません。'
               'Jgap≥0は平方根の凹性による代数で、科学仮説の支持ではありません。4×30・2×60の両方で端点の平方和分解を監査済みです。'
               'P120も有限で真値ではなく、絶対偏り・RMSE・補正・被覆改善・普遍的最適配分を示す実験ではありません。')


def _pooled(rows,seed_rows,decompositions,state):
    for field,ylabel in (('M','平均T（M）と原標本単位99.5%近似区間'),('Q','予算反復分散Qの原標本平均と99.5%近似区間')):
        fig=go.Figure()
        for dist,color in (('gamma','#46DCC0'),('lognormal','#FFB86B')):
            selected=[next(row for row in rows if row['distribution']==dist and row['method']==state['method'] and row['allocation']==name) for name in ALLOCATIONS]
            means=np.array([float(row[field]) for row in selected]);low=np.array([float(row[field+'_decision_low']) for row in selected]);high=np.array([float(row[field+'_decision_high']) for row in selected])
            bars=dict(type='data',symmetric=False,array=high-means,arrayminus=means-low) if np.isfinite(np.r_[means,low,high]).all() else None
            fig.add_scatter(x=list(ALLOCATIONS),y=means,mode='lines+markers',name=LABELS[dist],line_color=color,error_y=bars)
        _chart(fig,'同一120系列予算の配分（選択方法、P60は記述のみ）',ylabel)
    st.caption('tはM/Q=0の点のみを表示し、SE0の比較区間はNAです。他の非有限区間も描かず、保存表のNAを保持します。'
               '3配分を並べた区間は記述用で、確認的判断は後段のBCa・P120対P30の6区間だけです。')
    compact=['distribution','allocation','J','K','method','original_total','M','M_mcse','Q','Q_mcse','Q_over_R','Vbar','T','C','miss_T','warning_rate','interval_count','invalid_count','missing_original_count']
    st.dataframe(_display([{key:row[key] for key in compact} for row in rows]),hide_index=True,width='stretch')
    st.info('B=4999は1系列の再標本数、Kは1測定の系列数、Jは1予算で平均する測定数、JK=120です。R=20は予算全体の反復、N=30は各分布の独立原標本数です。'
            'M07でJ20と呼んだ役割を今回はRと呼びます。3配分の比較で生成数を3倍に数えず、同一予算は再標本化の仕事量であって同一wall時間の保証ではありません。')
    st.info('各seed N10、各分布N30、計60原標本はM07/M06/M05保存標本の固定ID再利用です。新しい原標本による独立追試ではありません。'
            'R/J/KをNに足しません。有限R/Kノイズを含む原標本間のモデルベース近似区間で、固定60標本を条件とした内側乱数だけの厳密区間ではありません。'
            'N30・非正規な原標本別量・既知結果を見た設計という限界があります。C/方向別ミス/幅/警告は記述値で、M05〜M07の既存判断を更新・再検定しません。')
    with st.expander('保存済みプール全18行（全分位点・conditional_・分母）'):st.dataframe(_display(rows),hide_index=True,width='stretch')
    with st.expander('保存済みseed別全54行（各N10・記述）'):st.dataframe(_display(seed_rows),hide_index=True,width='stretch')
    with st.expander('保存済み分解集計24行（全符号・残差・比）'):
        st.dataframe(_display(decompositions),hide_index=True,width='stretch')
        st.caption('mean_individual_Q120_Q30は個別比の平均、ratio_mean_Q120_over_mean_Q30は平均Qの比です。0分母はNA。conditional_は定義できた完全例の参考で、主判断に使いません。')


def _hypotheses(saved,decompositions):
    st.subheader('4. 事前固定したBCaの主比較（計6本・各分布N30）')
    quantities=saved['quantities'];fig=go.Figure();records=[]
    for dist in LABELS:
        row=quantities['H1_'+dist]
        if row['valid']:fig.add_scatter(x=[row['mean']],y=[LABELS[dist]],name=LABELS[dist],mode='markers',error_x=dict(type='data',symmetric=False,array=[row['high']-row['mean']],arrayminus=[row['mean']-row['low']]))
        records.append(dict(母集団=LABELS[dist],N=row['N'],mean_d=row['mean'],MCSE=row['mcse'],lower995=row['low'],upper995=row['high'],判定=row['status']))
    fig.add_vrect(x0=-.0025,x1=.0025,fillcolor='#46DCC0',opacity=.15,line_width=0,annotation_text='±.0025 Ase単位');fig.add_vline(x=0,line_dash='dot',line_color='white')
    _chart(fig,'d=mean_r(T_P120−T_P30) の原標本平均（Ase単位）','H1：99.5%原標本t区間',280);st.dataframe(records,hide_index=True,width='stretch')
    fig=go.Figure();records=[]
    for dist in LABELS:
        for name in ('H2_L','H2_U'):
            row=quantities[name+'_'+dist]
            if row['valid']:fig.add_scatter(x=[row['mean']],y=[LABELS[dist]+' · '+name[-1]],name=dist+' '+name[-1],mode='markers',marker=dict(symbol='circle' if name=='H2_L' else 'diamond'),error_x=dict(type='data',symmetric=False,array=[row['high']-row['mean']],arrayminus=[row['mean']-row['low']]))
            records.append(dict(母集団=LABELS[dist],量=name[-1],N=row['N'],mean=row['mean'],MCSE=row['mcse'],lower995=row['low'],upper995=row['high']))
    fig.add_vline(x=0,line_dash='dot',line_color='white');_chart(fig,'L=Q120−.8Q30 ／ U=Q120−1.25Q30（Ase²、反復分散）','H2：4本の99.5%原標本t区間',340);st.dataframe(records,hide_index=True,width='stretch')
    ratiofig=go.Figure();ratio_rows=[]
    for dist in LABELS:
        row=next(row for row in decompositions if row['distribution']==dist and row['master_seed']=='pooled' and row['method']=='bca')
        ratio=float(row['ratio_mean_Q120_over_mean_Q30'])
        if np.isfinite(ratio):ratiofig.add_scatter(x=[ratio],y=[LABELS[dist]],name=LABELS[dist],mode='markers',marker=dict(size=10))
        ratio_rows.append(dict(母集団=LABELS[dist],平均Qの比=row['ratio_mean_Q120_over_mean_Q30'],個別比の平均=row['mean_individual_Q120_Q30'],mean_Q30=saved['population_decisions'][dist]['H2']['mean_Q30'],H2=saved['population_decisions'][dist]['H2']['status']))
    ratiofig.add_vrect(x0=.8,x1=1.25,fillcolor='#A8A2FF',opacity=.15,line_width=0,annotation_text='乗法帯 (.8,1.25)');ratiofig.add_vline(x=1,line_dash='dot',line_color='white')
    _chart(ratiofig,'記述点のみ：mean(Q120)/mean(Q30)（比の区間は計算していません）','H2の乗法帯と参考点推定',280);st.dataframe(_display(ratio_rows),hide_index=True,width='stretch')
    for H in ('H1','H2'):
        status=saved['hypotheses'][H]['status'];text={'supported':'支持（両分布が支持）','not_supported':'不支持（少なくとも一方が不支持）','inconclusive':'保留'}[status];st.write(f'{H}全体：{text}。')
    st.caption('H1：区間が厳密に±.0025内なら支持、完全に帯外なら不支持、接触・交差は保留。.0025はAse単位で、被覆率0.25ppではありません。'
               'H2：平均Q30が正・有限で、Lの下限>0かつUの上限<0なら支持。Lの上限≤0またはUの下限≥0なら不支持、それ以外は保留です。'
               '必要な両区間がNA/SE0なら保留。(.8,1.25)は乗法的な反復分散の同等帯で、「±25%」やSDの帯ではありません。比の平均ではなく平均Qの比が対象です。'
               '比の点に信頼区間を付けておらず、Qの上下限を割って比の区間を作りません。判定は保存済みL/Uの区間によります。')
    st.caption('各仮説は両分布支持で全体支持、いずれか不支持なら全体不支持、その他保留。P60・seed別は記述のみで追加ゲートはありません。'
               '平均と反復分散の両方の同等性は両仮説・両分布の支持に限定します。6本の99.5%近似区間の和事象上界は名目約3%以下を意図しますが、近似の妥当性が前提で厳密保証ではありません。'
               '同等性は普遍的最適配分、個別原標本の保証、独立な新原標本での追試を意味しません。')


def _sample_csv(sample,state,selection,digest):
    buffer=io.StringIO();writer=csv.writer(buffer);writer.writerow(['distribution','master_seed','outer_id','observation_index','X','sample_sha256','source_question'])
    for index,value in enumerate(sample):writer.writerow([state['distribution'],selection['seed'],selection['outer'],index,float(value),digest,'RQ-M07 / RQ-M06 / RQ-M05'])
    return buffer.getvalue().encode()


def _endpoint_csv(panel,state,selection,digest):
    buffer=io.StringIO();fields=['distribution','master_seed','outer_id','budget_repeat_id','stream_id','method','B','sample_sha256','lower','upper','valid','reason','a','z0','q_low','q_high','tau_low','tau_high','tail_warning']
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();oi=selection['outer']
    for rep in range(20):
        for mi,method in enumerate(METHODS):
            for stream in range(120):
                row=dict(distribution=state['distribution'],master_seed=selection['seed'],outer_id=oi,budget_repeat_id=rep,stream_id=stream,method=method,B=4999,sample_sha256=digest,reason=str(panel['reason_labels'][panel['reason_codes'][oi,rep,mi,stream]]))
                row.update({key:panel[key][oi,rep,mi,stream].item() for key in fields[8:] if key!='reason'});writer.writerow(row)
    return buffer.getvalue().encode()


def render_allocation_view():
    st.header('Inference Lab · 同じ予算で、どう測る？')
    st.caption('保存結果専用 · n20・固定α=.05・B4999・R20・各分布N30。P30=J4×K30、P60=J2×K60、P120=J1×K120。画面操作で系列・原標本を生成しません。')
    state=_state();selection=state['selections'][state['distribution']]
    for key,value in dict(distribution=state['distribution'],allocation=state['allocation'],method=state['method'],**selection).items():st.session_state.setdefault(f'au_{key}',value)
    st.sidebar.selectbox('同一予算の母集団',list(LABELS),format_func=LABELS.get,key='au_distribution',on_change=_population)
    st.sidebar.selectbox('同一予算のmaster seed',[0,1,2],key='au_seed',on_change=_remember)
    st.sidebar.number_input('同一予算の原標本ID',0,9,step=1,key='au_outer',on_change=_remember)
    st.sidebar.selectbox('予算反復 r',list(range(20)),key='au_r',on_change=_remember)
    st.sidebar.selectbox('同一予算の配分',list(ALLOCATIONS),format_func=lambda name:f'{name} · J{ALLOCATIONS[name][0]}×K{ALLOCATIONS[name][1]}',key='au_allocation',on_change=_remember)
    st.sidebar.selectbox('同一予算の手法',list(METHODS),format_func=METHODS.get,key='au_method',on_change=_remember)
    st.sidebar.caption('B4999・α=.05・R20固定、JK=120。同じID/hashを配分・手法・r・Lab・モード切替で保持し、分布ごとseed／ID／rを復元します。分布変更は独立した別原標本です。')
    selection=state['selections'][state['distribution']]
    try:
        path=EVIDENCE/'panels'/f'{state["distribution"]}_seed{selection["seed"]}.npz';panel=_panel(str(path),path.stat().st_mtime_ns)
        names=('original_allocation_metrics.csv','original_comparison_metrics.csv','budget_metrics.csv.gz','group_metrics.csv.gz','decomposition_metrics.csv.gz','summary.csv','decomposition_summary.csv');tables={name:_load(name)[0] for name in names}
        pooled,pooled_bytes=_load('pooled_summary.csv');hypothesis_bytes=(EVIDENCE/'hypotheses.json').read_bytes();hypotheses=json.loads(hypothesis_bytes)
        sample=panel['samples'][selection['outer']];digest=hashlib.sha256(np.asarray(sample,dtype='<f8').tobytes()).hexdigest();hashes=[]
        for source in (M07,M06,M05):
            old=source/'panels'/path.name;hashes.append(_hashes(str(old),old.stat().st_mtime_ns)[selection['outer']])
        if not digest==panel['sample_sha256'][selection['outer']].decode('ascii')==hashes[0]==hashes[1]==hashes[2]:raise ValueError('saved M07/M06/M05 original provenance mismatch')
        if panel['samples'].shape!=(10,20) or panel['lower'].shape!=(10,20,3,120) or not panel['observed_streams'].all():raise ValueError('fixed panel shape/completeness differs')
        old_rows,_=_rows(str(M05/'pooled_summary.csv'),(M05/'pooled_summary.csv').stat().st_mtime_ns)
    except (OSError,KeyError,ValueError,IndexError) as error:
        st.error(f'RQ-M08保存結果／M07・M06・M05入力対応を表示できません：{type(error).__name__}。証拠ファイルを確認してください。');return
    j,k=ALLOCATIONS[state['allocation']];st.success(f'表示条件：{LABELS[state["distribution"]]} ／ seed={selection["seed"]} ／ 原標本ID={selection["outer"]} ／ r={selection["r"]} ／ {state["allocation"]}=J{j}×K{k} ／ {METHODS[state["method"]]} ／ n20・固定α=.05・B4999・R20')
    st.caption(f'原標本SHA-256：{digest}。M07・M06・M05保存hashと同じouter IDで一致。M07由来の原標本だけを再利用し、旧端点系列は混ぜません。受入入力HEAD：{panel["input_accepted_head"].item()}')
    select=lambda rows:[row for row in rows if row['distribution']==state['distribution'] and int(row['master_seed'])==selection['seed'] and int(row['outer_id'])==selection['outer']]
    st.subheader('1. 共通120系列をどう分けるか');_intervals(panel,state,selection)
    st.subheader('2. R20回の予算推定値・7組の測定・代数的分解')
    _measurements(select(tables['original_allocation_metrics.csv']),select(tables['original_comparison_metrics.csv']),select(tables['budget_metrics.csv.gz']),select(tables['group_metrics.csv.gz']),select(tables['decomposition_metrics.csv.gz']),state,selection)
    st.subheader('3. 3配分×3方法×2分布の全18条件');_pooled(pooled,tables['summary.csv'],tables['decomposition_summary.csv'],state);_hypotheses(hypotheses,tables['decomposition_summary.csv'])
    with st.expander('選択rの端点・z0/a/q/tau診断（警告を含む保存値）'):
        oi,rep=selection['outer'],selection['r'];mi=list(METHODS).index(state['method']);records=[]
        for stream in range(1 if state['method']=='t' else 120):
            row=dict(method=state['method'],budget_repeat_id=rep,stream_id=stream,allocation_group=stream//k,reason=str(panel['reason_labels'][panel['reason_codes'][oi,rep,mi,stream]]))
            row.update({key:panel[key][oi,rep,mi,stream].item() for key in ('lower','upper','valid','a','z0','q_low','q_high','tau_low','tau_high','tail_warning')});records.append(row)
        st.dataframe(records,hide_index=True,width='stretch');st.caption('tau=B×min(q,1−q)。tau<10の警告を除外せず、無効理由・NA・計画分母を保持します。')
    with st.expander('旧M05 K1の安定性NAと既存結論を保持'):
        fields=['distribution','method','K','C','T','lower_sd','upper_sd','S','Ase','Apop'];st.dataframe(_display([{key:row[key] for key in fields} for row in old_rows if row['distribution']==state['distribution']]),hide_index=True,width='stretch')
        st.caption('M05 K1はtを含む安定性がNA。M08のtゼロ変動とは異なります。M06/M07の受入済み判断も更新しません。')
    prefix=f'rq_m08_{state["distribution"]}_seed{selection["seed"]}_outer{selection["outer"]}'
    st.download_button('同一予算プールCSV（18行）を保存',pooled_bytes,file_name='rq_m08_pooled_summary.csv',mime='text/csv',key='au_summary')
    st.download_button('主比較JSON（6区間・全判定）を保存',hypothesis_bytes,file_name='rq_m08_hypotheses.json',mime='application/json',key='au_hypotheses')
    st.download_button('固定原標本CSV（20観測）を保存',_sample_csv(sample,state,selection,digest),file_name=prefix+'_sample.csv',mime='text/csv',key='au_sample')
    st.download_button('共通120系列の全端点CSV（7200行）を保存',_endpoint_csv(panel,state,selection,digest),file_name=prefix+'_endpoints.csv',mime='text/csv',key='au_endpoints')
    html=EVIDENCE/'figures/overview.html'
    if html.exists():st.download_button('自己完結HTML概要を保存',html.read_bytes(),file_name='rq_m08_overview.html',mime='text/html',key='au_html')
    st.caption('端点CSVはR20×3方法×120系列の7200論理行で、配分ごとに重複保存しません。tはbroadcastされた同一参照で実質1区間／原標本です。'
               'HTMLはBCaの配分別平均M・Qの概要です。6本の主区間・全判定は画面／JSON、全18条件はCSVで確認できます。保存データの出力であり、実ブラウザの保存操作とは別の検証範囲です。')
