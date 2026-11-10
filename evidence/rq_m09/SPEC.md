## 状態・今回の受入

<date start="2026-11-10" /> 設計版 RQ-M09-v1。M08のOwner承認を受け、独立原標本による直接追試を次の単一RQに選定した。ここでは設計・Notion登録・実行指示作成まで。M09の原標本生成・実験・コード変更は未実施。実装担当は下記固定範囲を受領して実装から新PR提出まで進める。

<mention url="https://app.notion.com/p/411251c5f08581318281cf8c6fe453ae">MATHSCAPE｜数学を触れる実験室</mention> ／ <mention url="https://app.notion.com/p/413251c5f0858167a60bf67c33382e9e">RQ-M08｜同じ再標本化予算をどう配分するか：J×Kと平均Aseの再現性</mention> ／ [PR #8](https://github.com/Tone75/mathscape/pull/8)。

M08受入HEAD <code>85382fd3ae8966dcf197d4006de21ea6edf9ef9f</code>、科学成果 <code>b68026a5f938c10ad10d6f57e0d8f900dcda8348</code>、事前固定 <code>697c673f69a68f040d7d072ae3b90f356be8abd6</code>。052〜055 Done／Next=false。GitHub HEAD一致・Open／未mergeを照合。M08のH1/H2両分布支持と旧成果を保持し、再受入を開始条件にしない。Owner承認、外部科学レビュー、Owner個別操作、PR mergeは別の状態である。

## 科学レビュー・単一RQ

M08では同じ120系列の配分を比べ、平均移動±.0025 Ase単位、平均反復分散比(.8,1.25)の同等性が固定基準で支持された。ただしM06〜M08は同じ60原標本を継続利用した。新しい内側乱数だけでは新原標本への再現性を確かめていない。GのU区間上限−3.729004272334764e−8は境界に近いが、M08の事前判定を後から変更する理由にはしない。

<b>RQ-M09：同じ生成分布から新たに独立生成した原標本でも、同一120系列予算のP30（4×30）とP120（1×120）の平均Ase・平均反復分散の実用的同等性は、M08と同じ判定基準で再現するか。</b>

M08を見て選んだ直接追試であることを明記する。RやNを増やして旧パネルの境界を押し込む作業、閾値を広げる作業、旧結果との合算は行わない。新コホート1個による限定的な追試であり、広い分布族・普遍的最適配分・長期反復での再現確率を推定するものではない。

M09が保留なら「この固定予算では同等性を再確認できない」であり、非同等やM08の誤りと断定しない。不支持もP30/P120いずれかの普遍的優位を意味しない。M08とM09は別々の判定として並べる。追加コホートを自動実行しない。

## 固定実験

| 項目 | 固定値・役割 |
| --- | --- |
| 分布 | G=解析的標準化Gamma、L=解析的標準化lognormal。M05〜M08と同じ母平均0・母分散1・母歪度2 |
| 原標本 | <b>新規60個</b>。G/L×master seed0,1,2×outer0..9。各分布N30、各標本n20 |
| 区間 | α=.05、B4999、BCa主・percentile記述・t決定的対照 |
| 予算反復 | R20、r0..19。各rに共通120新系列k0..119 |
| P30 | J4×K30。0..29／30..59／60..89／90..119 |
| P60 | J2×K60。0..59／60..119。記述のみ |
| P120 | J1×K120。0..119 |
| 推論単位 | 新原標本。各分布N30／df29。J/K/R/seed数をNに加えない |

3配分で同じ120系列を共有し、配分ごとに生成しない。各配分は重複なしで全120系列を使用。総主生成60×20×120=144000系列、719856000再標本組、14397120000抽出indices、432000論理区間（t broadcast含む）。組A/V25200、T10800、原標本配分540、seed要約54、pooled18行。主予算はM08と同じ。系列数・再標本数・抽出数の同一性であり、同じwall時間ではない。

G/Lは別の独立群。seed/outer一致をG/Lの対応標本やM08/M09の対応標本と解釈しない。seedは乱数管理ブロックで、seed別N10は記述のみ。

## 新原標本の生成と事前固定の順序

[受入済み分布実装](https://github.com/Tone75/mathscape/blob/85382fd3ae8966dcf197d4006de21ea6edf9ef9f/mathscape/matched_moments.py)の解析的変換を保持する。標本内で平均・SDを推定して再標準化しない。

- G：y=Gamma(shape=1, scale=1)、x=y−1。
- L：v=2 sinh(asinh(1)/3)、t=1+v²、s=sqrt(log1p(v²))、m=sqrt(t)、d=m v。z=standard\_normal(20)、x=(exp(s z)−m)/d。
- 丸めた定数を使わず、既存式・float64演算順序を保全。分布ラベル・母真値を推論APIに渡さない。
- 全60原標本について変換前y/z、変換後x、解析定数、ID、RNGキー、float64 little-endian C-order bytes/hashを保存する。変換の独立監査は保存y/zから行い、新しい乱数を生成しない。

PCG64／SeedSequence、entropy=master seed。主namespace=<code>0x4D3039</code>、smoke=<code>0x534D3039</code>、G1/L2。キーは<b>以下を正確に区別</b>する。

```txt
原標本（既存generatorと同じ6要素形式）:
(namespace, distribution_id, 20, outer_id, 0, 0)

内側系列（M08と同じ7要素形式）:
(namespace, distribution_id, 20, outer_id, 1, budget_repeat_id, stream_id)
```

J/K/方法/実行順/batch/attempt番号をキーに入れない。旧namespace・smokeと重複しないキー登録を確認。ハッシュ不一致だけで統計的独立を証明したとはしない。

<b>事前commit前にM09主原標本を生成・表示・探索しない。</b> 先に固定するのは全ID、生成規則・キー・数理・判定・監査・guard・環境・schemaとコードhashであり、まだ存在しない新原標本hashを事前固定したと書かない。

事前commitを照合した主attempt内で全60原標本を1回生成し、全原標本とreceipt/hashを永続化・読み戻し確認してから内側系列へ進む。M08元原標本との取り違え・同一配列の混入は生成規則と保存hashで監査し、想定外の一致は交換せず停止。旧データを新コホートへ混入・旧未使用IDを結果を見て選別・外れ値除外・再抽選しない。

受入済みlinear分位点、ddof1、半数tie z0、delete-one jackknife a、q/tauと有効性規約を保持。qクリップ、fallback、適応B、警告除外なし。同じbootstrap平均列をpercentile/BCaで共有し、t/a/seは原標本ごとに計算。int64 indices、batch最大100万要素。

## 推定量・主仮説

```txt
se_i = sample_SD(x_i) / sqrt(20)
V_irpj = [var_k(lower) + var_k(upper)] / (2*se_i^2)   # ddof=1
A_irpj = sqrt(V_irpj)
T_irp = mean_j(A_irpj)
M_ip = mean_r(T_irp)
Q_ip = var_r(T_irp)                                # ddof=1

d_i = mean_r(T_ir,P120 - T_ir,P30)
L_i = Q_i,P120 - 0.8*Q_i,P30
U_i = Q_i,P120 - 1.25*Q_i,P30
```

Tは平方根の後に平均する。sqrt(mean V)への置換不可。Qは1予算のTの測り直し分散で、Q/R、端点分散V、原標本間分散と区別する。H2はmean(Q120)/mean(Q30)に対応し、個別比の平均・SD比・「±25%」と呼ばない。

主比較は<b>BCaのP120対P30のみ</b>。新コホートの各分布N30でd/L/Uそれぞれ、平均±t(.9975,29)×SD/√30の両側99.5%近似t区間、計6本。丸め前の保存値で判定する。

| 仮説 | 支持 | 不支持 | 保留 |
| --- | --- | --- | --- |
| H1：新コホートでも平均差が±.0025 Ase単位内 | d区間全体が厳密に(−.0025,+.0025)内 | lo>.0025 または hi\<−.0025 | 境界接触・交差・その他 |
| H2：新コホートでも平均反復分散比が(.8,1.25)内 | Lのlo>0 かつ Uのhi\<0 | Lのhi≤0 または Uのlo≥0 | その他 |

H2の平均Q30は正・有限が必要。境界規則もM08から変更しない。各仮説は両分布支持なら全体支持、どちらか不支持なら全体不支持、その他保留。「平均・分散の同等性を再現」と言えるのはH1/H2とも両分布支持の場合のみ。

P60、percentile、seed別、個別Q比、分解は記述。P60の主比較昇格やseed方向ゲート、事後の頑健性ゲートを追加しない。M08との区間・点推定の併記は記述のみで、2コホートをpoolせず、M08との独立2標本検定も追加しない。M08の既知結果をM09の判定入力にしない。

各方法で全R20×120端点・診断の完備性を判定する。必要欠測・非有限・se≤0は該当原標本主量NA、計画N30を縮めず該当分布主判定を保留。主比較SE0もNA/保留。完全例参考はconditional\_と明示し主結論に使わない。無効を0や非被覆へ置換しない。tのA/T/Q/dは厳密0、比・比較区間はNA。旧K1 NAを0に変更しない。

.0025と(.8,1.25)はM08から継承する設計上の実用幅で、公的標準・普遍的定数・検出力保証ではない。6本99.5%区間は近似が妥当な場合の名目多重性配慮で、N30・R20・歪みのもとで厳密被覆を保証しない。P120は有限の比較対象で真値ではない。絶対偏り・RMSE・被覆改善・全原標本での同等性は対象外。

## 保存と独立監査

全60の変換前後原標本・キー・hash、全144000系列の端点/診断、7組A/V、全T/M/Q/d/L/U、全6主区間と判定、18/54要約、分解残差、失敗/資源/環境を保存。M08の参照は固定SHA・hash付きで比較専用と明示する。

別実装は主集計・主判定関数をimportせず、全保存端点から全表・区間・判定を再計算。全60のy/z→解析的変換→xを独立照合。監査は主実験やRNGを呼ばない。原標本RNGの全出力を別PRNGで検証したとは主張しない。

指定36系列=G/L×3seed×outer0×r0×k\{0,29,30,59,60,119\}についてindicesとbootstrap平均列を保存し、別実装でindices→means→端点/診断を照合。残143964系列の生成過程完全監査、外部査読とは区別する。

全方法・原標本・rの4×30/2×60で端点平方和分解を確認。4×30では<br /><code>T120−T30 = (sqrt(mean_j V30)−mean_j sqrt(V30)) + (sqrt(V120)−sqrt(mean_j V30)) = Jgap+Pgap</code>。<br />Rgapなし。代数恒等式を科学仮説に数えない。主前に許容差・schema・全ID/キー・依存/source hash・資源guard・保存reserveを固定。許容差はM08の同種規約を継承し、必要差分は結果を見る前に根拠とともに記録する。

## 実行回数・資源・失敗

主1回：原標本生成・保存から外側wall300秒、Peak RSS2GiB、追加公開成果50MiB、batch100万要素。M08実績58.60秒／254918656 bytes／17.389749MiBは参考で、今回の実測や保証ではない。通常テスト/UI/監査から主を起動しない。主/smokeの起動担当は1名。

指定smoke1回は<b>M08保存原標本</b>G/L×3seed×outer0の6個を使い、別namespace、R2、共通12系列、B99、配分4×3/2×6/1×12（計144系列）。新M09主原標本を先に見る用途に使わない。原標本変換・キー検査は保存raw/決定的fixtureで事前検証する。追加のランダムpilotや検出力実験なし。

上限前に部分証拠を残して停止。不支持・保留・時間/RSS/容量超過・入力不一致を理由とする再抽選、自動縮小、条件変更は禁止。実コード不具合修正のみ、原因・失敗ソース/結果を保全した別attempt最大1回、主累計wall600秒以内を許す。修正再実行は同じ固定ID・キー・既生成の同じ原標本を使い、新コホートへ差し替えない。未生成分がある場合も同じ事前規則だけを使う。attempt間を合算して主結果を作らない。生成規則/母集団/主判定自体に実質変更が必要なら修正実行の対象外として停止し報告する。

## 保存ビュー・検証

Inference Labの「同じ予算で、どう測る？」に<b>旧パネルM08／新標本M09</b>を切り替える保存ビューを追加する。新しいモードやLabは増やさず、既存M08の利用を保全。「標本を変えても、同じ結論？」を入口に、平均差と許容帯、分散同等性の2境界、各コホートの支持/保留/不支持を並べる。初期表示では主BCa/P30対P120を明確にし、全条件・診断は詳細へ置く。

分布ごとのID復元をコホートごとに分離し、同じIDの旧新原標本を取り違えない。N/n/B/J/K/Rと有限Kの意味を必要箇所で説明。95%端点区間と99.5%比較区間を区別。全3手法・3配分・2分布にアクセスできる。操作で原標本生成/bootstrap/主実験をしない。P120真値線・最適配分ラベルなし。

全保存値と表示・CSV/JSON/自己完結HTML/原標本/選択端点の5種類のdownload内容を照合。既存4 Lab/M02〜M08、原標本hash、状態復元、自由探索αと固定α、旧K1 NAを保全。M08全1268pytestを基準に必要な数理/生成規則/監査/guard/AppTest、全pytest、pip check。試験件数を目標にせず意味のある回帰を追加する。

実ブラウザ利用可能なら新旧コホート・分布・方法・配分・ID/r・Lab/モード往復、代表download5種、HTML実表示を確認。不可なら設定変更/回避せず未検証を明示。AppTest/HTTP/bytesを実操作完了へ置換しない。旧全操作・別browser/mobile・Owner個別操作・外部科学レビューは実施した範囲だけ報告する。

## Git・実行順・提出

ローカルMATHSCAPEリポジトリでAGENTS/README/HANDOFF・Git状態・並行作業を確認。ユーザーの変更をreset/退避せず必要なら新branch/worktreeへ隔離。受入HEADを含む新branch、PR #8未mergeなら<code>feat/rq-m08-budget-allocation</code>をbaseに新stacked PR。merge済みなら受入内容を含む統合先を確認する。PR番号は事前に決めつけない。

056受領→057生成規則/数理/監査/guardの事前commit→058新60原標本＋主1回/独立監査→059保存ビュー/検証/新PR/同期。開始した票だけIn Progress、Next056のみ。evidence/rq\_m09へSPEC/manifest/CODEX\_RECEIPT/原標本receipt/全結果/監査/REPORT/validation/資源を保存。科学成果SHAと提出HEADを区別する。

通常commit/push・新PRをレビュー待ちで提出し、README/HANDOFFとNotion親子票/Home/仕様を同期。既存PR/main/タグ/Release/旧証拠・受入結論を保持。新成果のDone/Owner受入は代行しない。Notion不可時はPENDING\_NOTION\_SYNCを保存して受領仕様内作業を継続する。

公開Gitに秘密情報・個人メール・個人絶対パスを入れない。変更ファイルのローカルフルパス一覧はGit対象外に保存してOwnerへ渡す。他PJ・別RQ・依存追加/更新・課金・クラウドGPU・ホスティング・merge・新タグ・Releaseは対象外。実装計画だけで止めず、範囲内の検証・提出まで進める。

最終報告は仮説別/分布別の6区間と分母、18条件、旧M08と新M09の判定対比、独立性確保の手順と限界、監査範囲、主/smoke/再試行数・資源、事前固定/科学/提出SHA・新PR・Notion同期・未検証・次RQ候補1件を含める。保留でも自動増量せず停止。

## 一次資料と設計判断

- [M08固定REPORT](https://github.com/Tone75/mathscape/blob/85382fd3ae8966dcf197d4006de21ea6edf9ef9f/evidence/rq_m08/REPORT.md)／[主判定JSON](https://github.com/Tone75/mathscape/blob/85382fd3ae8966dcf197d4006de21ea6edf9ef9f/evidence/rq_m08/hypotheses.json)／[固定SPEC](https://github.com/Tone75/mathscape/blob/85382fd3ae8966dcf197d4006de21ea6edf9ef9f/evidence/rq_m08/SPEC.md)。
- [NumPy SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html)：再現可能な擬似乱数系列分離の実装参考。数学的な独立性の絶対保証と表現しない。
- [SciPy t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)：t分位点の実装参考。非正規量への近似の妥当性を保証する資料ではない。
- RQ・コホート規模・許容帯・停止規約は本設計による。公式資料が推奨する普遍的最適配分ではない。

## 実行票・依存

| 票 | 依存・役割 | 登録時状態 |
| --- | --- | --- |
| <mention url="https://app.notion.com/p/413251c5f085816b905bee190a0bd573">MATHSCAPE MATH-056｜RQ-M09｜独立原標本で同一予算配分の同等性を追試（実行親票）</mention> | 実行親票 | Ready／Next=true |
| <mention url="https://app.notion.com/p/413251c5f08581cda9a3c43dfa020d53">MATHSCAPE MATH-057｜RQ-M09｜生成規則・新コホート隔離・推定量と判定を事前固定</mention> | 056受領→生成規則・数理・監査の事前固定 | Ready／Next=false |
| <mention url="https://app.notion.com/p/413251c5f08581a5a643e9a382c644bb">MATHSCAPE MATH-058｜RQ-M09｜新60原標本・144000系列の固定追試と独立監査</mention> | 057事前commit→主1回・独立監査 | Backlog／Next=false |
| <mention url="https://app.notion.com/p/413251c5f08581b9b905e14df1e6c073">MATHSCAPE MATH-059｜RQ-M09｜旧新コホートの保存比較ビュー・検証・提出</mention> | 058保存成果→比較ビュー・検証・新PR・同期 | Backlog／Next=false |

## Codex実行プロンプト｜RQ-M09-v1

```txt
MATHSCAPEのRQ-M09を、実装・事前固定・固定実験・独立監査・保存ビュー・検証・新PR提出まで進めてください。

作業先：<local project directory>
固定仕様全文：
https://app.notion.com/p/413251c5f08581cbb609c8082d2f382e
実行親票MATH-056：
https://app.notion.com/p/413251c5f085816b905bee190a0bd573

最初に仕様全文・056〜059・Home・AGENTS・README/HANDOFF・Git状態と並行作業を確認し、CODEX_RECEIPTを残してください。以下は要約です。固定仕様の詳細条件を省略しないでください。

【受入と問い】
RQ-M08はOwner承認済み。受入HEADは85382fd3ae8966dcf197d4006de21ea6edf9ef9f、PR #8。052〜055 Done、M08の限定的なH1/H2支持と旧成果・未検証範囲を保持し、再承認を開始条件にしません。

問いは「新しい独立原標本でも、同一120系列予算のP30=4×30とP120=1×120の平均Ase・平均反復分散の同等性が、M08と同じ基準で再現するか」です。新コホート単独で判定し、M08とのpool・旧新IDによる対応比較をしないでください。保留は非同等やM08の誤りと断定しません。

【固定設計】
G/L×seed0,1,2×outer0〜9の新60原標本、各分布N30、n20。母分布は受入済みGamma/lognormalの解析的標準化を維持し、標本内再標準化・結果選別・再抽選なし。
α=.05、B4999、R20、各反復の共通120系列をP30=4×30、P60=2×60、P120=1×120で共有。主生成144000系列。BCa P120対P30のみ主比較、P60/percentile/seed別は記述、tは決定的対照です。

主namespace0x4D3039、smoke0x534D3039、PCG64/SeedSequence。原標本の6要素キーと内側の7要素キーを仕様どおり区別してください。
M09主原標本を生成・閲覧する前に、全ID・生成規則・キー・実装・判定・監査・guard・環境・schemaをcommit。未生成の原標本hashを事前固定済みとは書きません。
主attempt内で新原標本60個の変換前y/z・変換後x・キー・hashを全保存し、読み戻してから内側系列へ進んでください。

【推定量と仮説】
ddof=1で
V=(var(lower)+var(upper))/(2×原標本SE²)
A=√V、T_p=mean_j(A)、M_p=mean_r(T_p)、Q_p=var_r(T_p)
d_i=mean_r(T_P120−T_P30)
L_i=Q_P120−.8Q_P30
U_i=Q_P120−1.25Q_P30

Tをsqrt(mean V)へ置換しません。各分布N30/df29でd/L/Uの両側99.5%近似t区間6本を作り、丸め前値で判定してください。
H1：d区間全体が厳密に±.0025内なら支持。lo>.0025またはhi<−.0025なら不支持、その他保留。
H2：L下限>0かつU上限<0なら支持。L上限≤0またはU下限≥0なら不支持、その他保留。
各仮説は両分布支持で全体支持、片方不支持で全体不支持、その他保留。欠測・非有限・SE0は仕様のNA/保留規約を適用し、Nを縮めて支持を出しません。

H2は平均Qの比(.8,1.25)であり、個別比平均・SD比・「±25%」ではありません。J/K/RをNへ加えず、P120を真値・最適配分・被覆改善と解釈しないでください。M08のG_U境界近接と近似区間の限界を残します。

【実行・監査】
056受領→057事前commit→058主1回/独立監査→059保存ビュー/検証/提出。
主は原標本生成から外側wall300秒・RSS2GiB・追加公開成果50MiB・batch100万要素以内。起動担当は1名。
指定smoke1回のみ、M08保存6原標本×R2×12系列×B99を別namespaceで使用。M09主原標本の先読みには使いません。

不支持・保留・資源超過で追加抽選や自動縮小をしません。実コード不具合だけ仕様の別attempt最大1回・主累計600秒、同じ原標本/キーで修正実行可。失敗証拠を保全し、生成規則・母集団・主判定の実質変更が必要なら停止してください。

全60原標本の保存raw→解析変換、全保存端点からの全集計・6区間・判定を別実装で監査。主集計/判定関数を監査側にimportしません。
指定36系列（G/L×3seed×outer0×r0×k0,29,30,59,60,119）はindices→means→端点を復元。監査で乱数は再生成しません。残143964系列の生成過程完全監査とは区別します。
平方和分解・Jgap+Pgapを全件照合。Rgapなし。代数恒等式を科学仮説に数えません。

【UI・提出】
既存「同じ予算で、どう測る？」へM08/M09コホート切替を追加し、「標本を変えても、同じ結論？」として結果・許容帯を並べてください。新Lab/モードを増やさず、保存UIから実験を起動しません。旧新コホート別・分布別のID復元とhashを保全します。
既存4 Lab/M02〜M08、1268pytestを基準に必要な追加検証・全pytest・pip check、表示/保存値/5種類downloadを照合。実ブラウザ可能なら代表操作・実download・HTML表示、不可なら未検証として記録し、設定を回避しません。

受入HEADを含む新branchで作業。PR #8未mergeならfeat/rq-m08-budget-allocationをbaseに新stacked PRを通常push・提出してください。既存PR/main/タグ/Release/旧証拠を保持し、mergeはしません。
evidence/rq_m09、README/HANDOFF、Notion親子票/Home/仕様へ同期。公開Gitに秘密情報・個人絶対パスを入れず、変更フルパス一覧はGit対象外に保存。Notion不可時はPENDING_NOTION_SYNCを残して継続します。

開始した票だけIn Progress、Next056のみ。新成果はOwnerレビュー待ちで提出し、Doneを代行しません。6区間・全18条件・旧新の判定・監査範囲・回数/資源・事前固定/科学/提出SHA・PR・未検証・次RQ候補1件を報告し、追加実験せず停止してください。
```
