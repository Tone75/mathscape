# RQ-M11 保存ビュー契約（解析前固定）

対象は受入済みM10の保存n5原標本60個・各2400系列。M11はM10の保存解析であり、新コホート／新Lab／新しいトップレベル推論モードにしない。実M11の列挙・参照・誤差・判定をこの文書の作成時には計算しない。

## 保存値と表示の対応

- 手法軸は `bca, percentile`、端点軸は `lower, upper`。保存系列IDは `r*120+k`、r=0..19、k=0..119。3配分で重複計上しない。tはM10の既存表示を保持し、全列挙誤差の判定を追加しない。
- `references/{distribution}_seed{seed}_outer{outer}.npz` は順序付きindices `(3125,5)`、float64 ordered means、支持点・整数頻度・累積頻度、両手法・両端点のq／逆CDF参照／linear参照／CDF境界／tail診断を保存する。
- `comparisons/` の同名NPZは2400系列の保存端点、正規化差、linear参照との差、規約差、恒等式残差、q／z0／a、valid／reason／warning等を保存する。UIはこれらの値を読み、誤差・参照・集計を再計算しない。
- `reference_metrics.csv`／`endpoint_metrics.csv` は各240行、`original_metrics.csv` は120行、`summary.csv` は4行。全原標本・両手法を詳細表から確認できる。`hypotheses.json` はH11-Eだけ、`representatives.json` はG/L各群BCaの最小／最大e例（同点seed,outer昇順）を固定する。
- UIの合格数、計画N30、無効／欠測、e、余裕 `.10-e`、上下のb／sqrt(v)／RMSE／MAE／絶対差分位点は保存値を表示する。27/30以上かつ全必要値完備という保存判定を使い、表示丸め・数値許容差・conditional参考で合格数を変えない。
- 主解析未完了、入力hash不整合、独立監査未完了／不一致は、科学判定を未検証・保留として表示する。NAを0で補わず、群の計画分母を縮めない。

## 導線と互換性

`allocation_ui.py` のM10選択時だけ、既存配分ビューと「全列挙参照とのずれ（M11）」の詳細選択を追加する。既定は既存ビューを維持し、M11の入口とG/L合格数／30・0.10SE基準を短い保存結果プレビューで見つけやすくする。プレビューに新しいmetric／dataframe／plotを挿入せず、旧テストの表示順・旧5出力を維持する。

M11選択時は新しい `mathscape/enumeration_ui.py` に分岐し、既存ビューと同時に10個のdownloadを置かない。コホート・分布別のM10 seed／outer／r／hashを保持し、M11の手法選択はBCa／percentileの専用状態とする。M11から戻っても旧配分／手法／ID状態、M08/M09、既存4Lab、旧NA、自由探索αと固定αを保全する。

初期詳細はG/Lの合格数／30と設計閾値、選択原標本の上下端点の保存2400系列分布・逆CDF参照線・linear参照線・RMSEを表示する。技術詳細・全ID表・CDF／tie／整数境界・規約差は展開部へ置く。最小／最大例は保存された代表例だけを選び直す。

説明は「全列挙参照からの計算端点のずれ」。0.10SEを推定精度10%や母平均の誤差保証と呼ばない。参照は保存binary64・M10平均規約に対する条件付き有限分布で、実数真値／必ず到達するB→∞の極限ではない。M10保留の因果的説明・被覆改善・配分の優劣・母集団への90%保証を追加しない。

有限Bの端点ごとのtail診断 `B*min(q_e,1-q_e)` と、全列挙の `3125*min(q*_e,1-q*_e)<10` を区別する。q=0/1・少ないtailは有効なら警告であり、除外しない。UIでtailやqの意味を置き換えたり、新しい判定ゲートを作らない。

## M11の5種類の出力

選択標本の2出力にdistribution／seed／outer／sample SHA、n5、B4999、R20×120、手法軸・端点軸・参照規約の由来を付ける。全体要約と判定は保存された全体scopeのまま配信し、選択IDを混ぜない。CSVはUTF-8・安定した列順・binary64 round trip可能な表記を使い、未知／無効は空欄等の明示NAと理由を保持する。JSONは保存bytes、非有限値は標準JSONのnullである。

| 種類 | ファイル名 | 出力契約 |
| --- | --- | --- |
| 保存要約CSV | `rq_m11_summary.csv` | `summary.csv` の保存bytesをそのまま配信。全4行、BCaが唯一の主判定、percentileは記述対照。 |
| 保存判定JSON | `rq_m11_hypotheses.json` | `hypotheses.json` の保存bytes。群別／全体H11-E、計画N、合格数、無効・欠測を表示と照合。独立監査の状態は別の保存記録で確認し、未検証時のUI保留と生JSONの保存値を区別する。 |
| 選択標本の全列挙参照CSV | `rq_m11_{distribution}_seed{seed}_outer{outer}_reference.csv` | 保存参照NPZの3125 ordered means・添字、支持点／整数頻度／CDF、両手法・上下参照と診断を無損失に表形式化。異種行は `record_type=enumeration/support/reference` で区別する。参照を再計算しない。 |
| 選択標本の2400系列比較CSV | `rq_m11_{distribution}_seed{seed}_outer{outer}_comparison.csv` | 1系列1行、計2400行。series ID／r／k、両手法・両端点の保存値・差・linear成分・規約差・残差・valid/reason/warningと保存診断をwide形式化。配分で3倍にしない。 |
| 自己完結HTML | `rq_m11_overview.html` | `figures/overview.html` の保存bytes。保存G/L合格数・全ID eと0.10SE線、固定代表例の参照と保存端点分布を含む。表示範囲を明示し、全144000系列を図示したとは書かない。 |

参照／系列CSVは保存配列の整形のみで生成できる。全60個分のCSVを重複保存する必要はなく、選択時の純粋なserializationは科学runに数えない。元NPZとCSV全値を独立に照合し、byte決定性をテストする。

HTMLは主解析runnerで作らず、MATH-067開始後にUI担当が保存出力だけから一度作成する。小さな再利用可能writerを用い、Plotlyならlibraryを埋め込み、外部script／stylesheet／ネットワークを要求しない。HTMLの数値計算は行わず、保存された代表例・指標・分類を使用する。Rootが実ブラウザ・実download・HTML表示の確認を担当する。

## 検証契約

- 決定的な小さい保存fixtureで有効／無効／欠測／閾値接触、CDF／linear参照の区別、手法別無効伝播、代表例選択、NAと理由、5 exportの全列・byte決定性を検証する。
- AppTestは列挙・逆CDF・BCa・誤差集計・判定・bootstrap・RNG・主／監査runnerを禁止し、保存値だけの表示と状態復元を検証する。M11実データの主解析・全列挙を通常テストから起動しない。
- 旧M10の5 download、最初の保存原標本plot、6 metric、全表順序と旧M08/M09の代表導線を回帰確認する。新しいトップレベルモード／Labは増やさない。
- 監査成功後の実ブラウザではG/L、固定最小／最大例、任意ID復元、両手法、参照規約表示、5実downloadとHTML、旧3コホート／4Labの代表往復を確認する。AppTest／HTTP／byte照合を実操作完了とは呼ばない。
