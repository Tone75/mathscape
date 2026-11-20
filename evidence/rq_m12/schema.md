# RQ-M12 保存schema・数値契約（解析前固定）

実M12のq写像・g／r／uを計算する前に固定する。受入M11 HEADは`0deb14fa313dc27380afbb43e6b132386aa40206`。対象はM10由来n5原標本G/L×seed0..2×outer0..9の全60個、2400系列／標本。新標本・新乱数・PRNG再生・M11主解析再実行は0。旧入力・旧成果は変更しない。

## 主workerと全入力障壁

`python scripts/analyze_probability_movement.py --root <repo> --output <evidence-dir>`。root省略時はこのscriptのrepo、output省略時はroot/evidence/rq_m12。`--help`は解析を起動しない。

外側launcherが全入力Git blob/hash・受入HEAD・固定source／環境を確認し、outputの`manifest.json`と`input_receipt.json`を先に作成する。workerはそれらの存在を要求し、独自のGit検証器や別の科学起動経路を追加しない。`analyze(root,output)`は全60入力のキー・軸・dtype・ID/hash・M10/M11保存値の橋渡しを検証してから最初のq写像を行う。既存output内のSPEC／manifest／receipt／attemptを許し、結果ファイル／seriesディレクトリはexclusive作成して上書きしない。

入力のM11手法順は`bca,percentile`、端点順は`lower,upper`、系列順は`r*120+k`（r0..19,k0..119）。M10手法順`t,percentile,bca`からの投影を照合する。表示丸め値・CSVのqは使わず、NPZのbinary64確率を使用する。

## 数値規約

CDF支持点は有限・厳密昇順float64、frequencyとcumulativeは正整数頻度に対応するint64、総頻度3125。±0はM11の数値的同一支持点、近接値は丸めて統合しない。primaryは各qの`as_integer_ratio()`の分子num／分母denを用い、最初に`cumulative[j]*den >= 3125*num`となる支持点を二分探索する。q0は最小・q1は最大。独立監査はソート済み3125平均の整数rankで同値を照合する。

A=Q(q*)がM11のE_cdfとatom_indexへ厳密一致することを要求する。C=Q(q_B)、Y=保存有限B端点、S=M11標本SEで、g=(C-A)/S、h=(Y-C)/S、d=(Y-A)/S。dはM11保存deltaへ固定数値許容差で直接照合する。percentileは同じ手順のg=0を実装確認とし、独立の科学仮説にしない。percentileの必要無効をBCaへ伝播させない。

BCaの支持点分類はindexの符号差を使い厳密に判定する。q_changedはbinary64の`q_B != q*`、q_changed_sameはq_changedかつ支持点同一。別の近接支持点への移動をallcloseで「同じ」に変えない。距離は標本SE単位のabs(g)で、moved_abs_g_meanは実際に移動した系列だけの平均。移動0件ではNA。

入力identity／軸／dtype／受入保存値橋渡しの矛盾は停止対象。数理的無効・必要欠測は系列／端点ごとのNAと理由として保持する。優先順は未観測、参照無効、SE不正、M10保存無効、q_B非有限・範囲外・順序不正、Y非有限・順序不正、M11保存総ずれ無効、導出非有限。qの順序は0<=q_lower<q_upper<=1。下上確率の順序不正は両端を無効にする。M10有効warningを除外せず、q0/1も有効なら含める。クリップ／補充／fallbackはない。

## `series/{distribution}_seed{seed}_outer{outer}.npz` 全60ファイル

metadataはdistribution Unicode scalar、master_seed／outer_id integer scalar、sample_sha256 Unicode scalar、question_id=RQ-M12、source_question_id=RQ-M11、n5／B4999／R20／K120。endpoints Unicode(2)、budget_repeat_id／stream_id uint8(2400)がr-majorを記録する。object配列は使わない。

| field | shape／dtype | 意味 |
| --- | --- | --- |
| q_B, q_delta | (2400,2) float64 | 保存BCa確率、q_B−q* |
| q_star, A | (2) float64 | M11保存確率、同じCDFの基準端点 |
| A_index | (2) int32 | 基準支持点index（0始まり）、未知−1 |
| C, Y | (2400,2) float64 | q_Bで選ぶ同じCDFの端点、保存有限B端点 |
| g, h, d, abs_g | (2400,2) float64 | 主移動・残差・総ずれ・絶対移動（SE単位） |
| selected_index | (2400,2) int32 | Cの支持点index、未知−1 |
| valid | (2400,2) bool | 全必要量が既知・有効 |
| reason_code | (2400,2) uint8 | reason_labelsへのindex、0は有効 |
| reason_labels | (可変) Unicode U200 | 保存無効理由を含む辞書 |
| movement_code | (2400,2) int8 | −2未知、−1下、0同じ、+1上 |
| support_step | (2400,2) int32 | 選択index−基準index、未知はint32最小値 |
| q_changed, q_changed_same | (2400,2) bool | 無効時はFalse、validと併読し未知を同一点に数えない |
| endpoint_identity_residual | (2400,2) float64 | d−(g+h) |
| total_error_residual | (2400,2) float64 | 直接d−M11保存delta |
| se | scalar float64 | M11の原標本SE |
| observed | (2400) bool | 保存系列の観測有無 |
| source_warning | (2400) bool | M10のBCa method-wide warning（除外しない） |
| source_tail_count | (2400,2) float64 | M10保存tau_lower／tau_upper |
| support, frequency, cumulative | (支持点数) float64／int64／int64 | 固定M11 CDF |
| cdf_probability | (支持点数) float64 | 保存累積頻度/3125、表示用に一度保存 |
| percentile_q_B, percentile_C, percentile_g | (2400,2) float64 | 固定確率対照（新判定なし） |
| percentile_q_star, percentile_A | (2) float64 | 対照の基準 |
| percentile_selected_index | (2400,2) int32 | 対照選択index |
| percentile_reference_index | (2) int32 | 対照基準index |
| percentile_valid | (2400,2) bool | 対照有効性、BCaと独立 |
| percentile_reason_code | (2400,2) uint8 | 同じreason_labelsへ接続 |

無効のC／g／h／d／q_deltaはNaN、movement_codeは−2。元のq_B／Yと理由・観測maskは保持する。NaNを0移動へ置換しない。CSV未知値は明示NA、JSON非有限値は標準nullに正規化する。

## 表と判定

| file | 計画行数 | 内容 |
| --- | --- | --- |
| original_metrics.csv | 60 | ID/hash、u、余裕.10−u、pass_flag、complete／invalid／missing／reason、下上r、rank_u、M11既存e／pass／complete、元warning系列数 |
| endpoint_metrics.csv | 120 | ID/hash／端点、計画2400／観測／有効／無効／欠測、same/down/up/moved件数、q_changed／q_changed_same件数・率、g符号付きmean／RMS／abs p95/max／全系列MAE、移動時だけの平均abs(g)、q差のmean/min/max/MAE/abs p95/max、h／d mean/RMS、二乗平均と交差項・恒等式、percentile g0対照件数 |
| summary.csv | 4 | G/L×lower/upper、計画原標本30／完備数／無効欠測数、原標本別率・量の等重み平均、rのlinear p50/p95/max、same/down/up等の補助総件数 |
| classification.csv | 8 | 群ごとのM11合否×M12合否の4cells、countと`seed:outer`の昇順ID列（セミコロン区切り） |
| classification_ids.csv | 60 | 全IDのM11/M12 nullable pass、category=00/01/10/11またはunknown、理由 |
| reasons.csv | 可変 | ID/hash／端点／valid・invalid・missing／理由／系列数。非空の各理由を残す |
| decision.json | 1object | 唯一のH12-Q、G/L群別／全体computed_status・status、計画30と27条件・無効欠測不明数、分類8cells、分類unknown IDs、verification_required=true |
| representatives.json | 1object | G/L各群u最小／最大、同点seed・outer昇順、保存ID/hash/u/余裕/rank |
| analysis_resources.json | 1object | 内部ループ時間、完了ID／系列・BCa端点数、new_originals/bootstrap/PRNG/M11再実行0、永続化まで成功時だけcompleted |

主rはsqrt(mean(g²))、u=max(r_lower,r_upper)。全30必要標本が完備ならu<=.10が27以上で記述的支持、26以下で記述的不支持。必要無効／欠測群は計画30を保持して保留。監査未完了・不一致時は公開結論を未検証として保留し、computed値だけを受入済みと表示しない。

endpoint_metricsのprimary率／量は2400必要系列が全て有効な場合だけ定義する。既知件数は欠測時も保持し、conditional_参考は既知有効系列を分母として保存するが主判定に使わない。same_rateはsame_count/2400、q_changed_same_rateはq_changed_same_count/2400。same_given_q_changed_rateはq_changed_same_count/q_changed_countで、q変更0件ならNA。移動時平均も移動0件ならNA。

summaryは原標本ごとに集計した指標の等重み要約で、2400系列を独立原標本Nへ加えない。全30が不完備ならprimary summaryの量はNA、conditional_完全例参考を区別する。分母が0で定義できない原標本条件付き率やmoved_abs_g_meanは、groupでも定義数を併記し、その定義された原標本だけの記述平均とする。新たなCI／p値／支持判定は追加しない。

## 恒等式と許容差

d=g+hとmean(d²)=mean(g²)+mean(h²)+2mean(g*h)を直接照合し、交差項の符号を保存する。算術量の照合はrtol2e−10／atol2e−12、恒等式rtol=atol=2e−12。point scale=max(1,abs(g),abs(h),abs(d))、moment scale=max(abs(mean(d²)),abs(mean(g²)),abs(mean(h²)),abs(交差項),abs(成分和))で、交差項の相殺を隠して0扱いしない。各残差・scale／正規化を保存し、有限値の恒等式破綻は実装エラーとして停止する。導出overflowはNA／理由にしcompleteにしない。

支持点同一／上下・ID/hash・分類・順位は厳密一致、監査許容差や表示丸めで動いた点を同一点／uを合格へ変更しない。hは経験分布と補間規約の差をともに含み、同一系列のg/hは依存し得る。因果寄与率を出さない。
