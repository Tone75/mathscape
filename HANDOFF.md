# MATHSCAPE — minimal handoff

## 最新の受入と実行状態：RQ-M08

Owner受入済みM07 HEAD `39bcdc367320b6da513df72d2fa43bfdb7f060f4`、PR #7、048〜051 Doneを採用。M07の限定H1/H2支持と旧結論・実ブラウザ等の未検証を保持し、再承認を開始条件にしない。開始tree clean・別worktreeなし、新branch `feat/rq-m08-budget-allocation`。固定仕様全文・末尾プロンプト・052〜055・Home・AGENTS有無・READMEを確認し、CODEX_RECEIPTを残した。開始時全1139pytest成功45.00秒、pip check成功。

052受領→053事前固定→054主1回／独立監査→055保存UI／検証／新PR。Next052のみ。60保存原標本×R20×共通120新系列を3配分P30=4×30/P60=2×60/P120=1×120で共有。T=mean_j√V、主d/L/Uは原標本N30／分布、99.5%区間計6本。平均差±.0025と分散比(.8,1.25)を別判定し、後者を±25%／SD比と呼ばない。新規原標本・旧端点混入・結果選別なし。通常テスト／UI／監査は主を起動せず、旧証拠を保全する。詳細は `evidence/rq_m08/SPEC.md`／`CODEX_RECEIPT.md`。新成果は別stacked PRへレビュー待ちで提出する。

事前固定 `697c673f69a68f040d7d072ae3b90f356be8abd6`、manifest SHA256 `ee870c8f1c13ada089c0556b72baacdaceb3c2bed13b35eca8d9acb69e3420d9`。指定smoke1回／主1回／再実行0／原標本追加生成0。主外側58.60秒／RSS254,918,656 bytes、無効・欠測0、H1/H2は両分布支持。GのH2 U上限−3.7290042723e−8は近接境界で、正確な保存数値を判定に使う。全18条件・全6主区間・平方和とJgap+Pgap、指定36系列の独立復元で不一致0。残143,964系列の生成過程完全監査ではなく、外部科学レビューでもない。

再開入口は `mathscape/allocation.py`／`allocation_ui.py`、`scripts/run_allocation_guarded.py`／`run_allocation_benchmark.py`／`audit_allocation.py`。詳細は `evidence/rq_m08/REPORT.md`。数理／runner／監査の18事前固定ファイルと21旧入力を保持し、通常テスト／保存UI／監査は主を起動しない。新成果はレビュー待ち、052〜055 In Progress・Next052のみ。旧受入と実ブラウザ等の未検証状態を保持し、追加実験しない。

全1,268 pytest成功（63.83秒、新AppTest15）／pip check成功。実ブラウザで両分布／3配分／全方法・ID／r／Lab／モード往復／固定αを確認、18状態108数値一致、5種類の実ダウンロードとHTML実表示を完了。新旧代表操作の範囲・未検証は `evidence/rq_m08/browser/verification.json`／`validation.json`。旧全操作・別browser／モバイル・Owner個別操作・外部科学レビューは未実施。旧M06/M07未検証は履歴として保持する。取得console error/warning0、対象重複初期値警告なし、確認用2サーバー／2tab終了済み。

## 最新の受入と実行状態：RQ-M07

Owner承認済みM06の受入HEAD `e2feae23397d014f6aa3363773da9913da85fcf7`、PR #6、044〜047 Doneを採用。有限K30の結論とM05以前を保持し、実ブラウザ等の旧未検証を検証済みに変えない。旧レビュー待ちは開始条件ではない。開始tree clean・別worktreeなし、新branch `feat/rq-m07-finite-k`。仕様全文・末尾プロンプト・048〜051／Home・AGENTS有無・READMEを確認し、CODEX_RECEIPTを記録した。

048受領→049事前固定→050主1回／独立監査→051保存結果UI／検証／新PR。Nextは048のみ。M06保存G/L×3seed×outer0..9の60原標本、B4999・J20・K120と先頭K30、原標本単位N30／分布。J/P/R三項と端点平方和を照合し、K120真値・絶対偏り補正を主張しない。通常テスト・UI・保存監査は主を起動せず、旧証拠は保全する。詳細は `evidence/rq_m07/SPEC.md`／`CODEX_RECEIPT.md`。新成果はレビュー待ちとして別stacked PRへ提出する。

事前固定 `12b4aa79bb47e196006538d7c0cf0d073ccb3c37`、manifest SHA256 `25ffd0b68050fed1f755463a4dc929a86447bb0b5c3e491002edd01378d976ab`。指定smoke1回・主1回・再実行0、原標本追加生成0／無効・欠測0。各N30/df29、H1はG/Lとも平均d区間が厳密に±.0025内で支持、H2は両平均W区間上限<0で支持。主外側53.06秒／RSS255,574,016 bytes。全432,000論理区間・全表・4主区間と指定24系列の独立復元で不一致0、残143,976生成過程は完全復元していない。K120真値化・絶対偏り・被覆改善・Jgapを新仮説の支持とする説明はしない。

保存UI・検証は `mathscape/finite_k_ui.py` と `tests/test_finite_k_app.py`、数理／guard／監査は `finite_k.py`／`scripts/run_finite_k_guarded.py`／`run_finite_k_benchmark.py`／`audit_finite_k.py` が入口。実ブラウザ可否は今回もWork設定disabled、実操作・実ダウンロード・HTML実表示は未検証。HTTP／health／限定ログ・HTML静的値とは区別し、旧M06未検証も保持する。確認用2サーバーは停止済み。通常pytest／保存UI／監査は主を起動しない。新成果はレビュー待ち、Next048のみ、追加実験なし。

全1,139 pytest成功（40.94秒）、新AppTest17件、pip check成功。保存UIの全12/36/24表・4主区間・原標本／状態保持・5 payloadを照合。実ブラウザ・実ダウンロード・HTML実表示は未検証のまま。詳細は `evidence/rq_m07/REPORT.md`／`validation.json`、次の候補は同じ再標本化予算でJ/K配分が平均Ase測定の再現性をどう変えるかの1件のみで未実行。17固定source・旧入力14ファイル・旧証拠を保持し、追加実験せずレビュー待ちで停止する。

提出先は [stacked PR #7](https://github.com/Tone75/mathscape/pull/7)、科学成果SHA `2532a8cc28911b1f320d4414621386fdd61f9955`、事前固定SHA `12b4aa79bb47e196006538d7c0cf0d073ccb3c37`。baseは受入済み `feat/rq-m06-matched-stability`。この参照追記は科学成果固定後の文書commitで、最新提出HEADは `gh pr view 7 --repo Tone75/mathscape --json headRefOid` で取得できる。Notion048〜051・Home・固定仕様へ結果／SHA／PR／検証範囲／資源／未解決事項を同期済み、PENDING_NOTION_SYNCなし。親票への一時HTTP502は未保存を再読確認し、その不足分だけ再送して解消した。新成果はレビュー待ちで048〜051 In Progress、Next048のみ。保護対象11 remote refsと旧証拠は開始時と一致、既存PR／main／タグ／Releaseを保持し、新規merge・タグ・Releaseは作成していない。

## 最新の受入と実行状態：RQ-M06

Owner明示承認と最新Notionに従い、RQ-M05（提出HEAD `dc2501c3099f5c25227d9eee00d227c2f0b02f81`、科学成果 `254af9ed19cf3197310bc372c704ecc32c3b9b4e`、事前固定 `a8e87b3deb1a5256a200b28ea51301dec8aecb7e`、PR #5）は受入済み、040〜043 Done。H1/H2実用差不支持・C/Tに限定した±1pp内同等性支持を保持する。旧レビュー待ちは提出時点の履歴であり開始条件ではない。merge・外部科学レビュー・Owner本人の個別新操作とは別。

CODEX_RECEIPT RQ-M06-v1：開始HEADは受入済みと一致、clean／別worktreeなし。新branch `feat/rq-m06-matched-stability`。仕様全文・末尾プロンプト・044〜047・Home、README／HANDOFF、M05入力と受入数理を確認。既存862 pytest成功（20.72秒）。開始した044／045 In Progress、Nextは044だけ。045事前固定→046主1回／独立監査→047保存結果UI／検証／提出の順で実行する。詳細は `evidence/rq_m06/SPEC.md`／`CODEX_RECEIPT.md`。

固定入力はM05のG/L×seed0,1,2×outer0〜199、全1,200原標本。入力hash／IDを事前固定し追加母標本生成なし。新namespaceでK30・B999/4999共有prefix、原標本単位N600/分布。有限K30のAseを比較し、M05 C/Tを再検定・プールしない。旧K1のNAは維持する。主1回・指定別namespace smoke1回のみ、通常pytest・UI・監査は主実験を起動しない。旧証拠／PR／main／タグ／Releaseを保持し、新成果は別stacked PRのレビュー待ちとして提出する。

事前固定 `3af12e48bec70381a4fc8d70b9b25c8b1e95c405`、manifest SHA256 `1fa89bbb0079e73ee7b3fcb993ca0fa0a336adc8ed8e6964ab8667a264077f58`。指定smoke1回／主1回、再実行0・母標本生成0・無効／欠測0。H1は.01超の実用差不支持かつ固定条件の平均Aseの±.01内同等性支持、99%区間[−.0043907952,−.0002664678]。H2は両分布25%超平均低減支持、平均比G .457544／L .461710。原標本単位N600／分布、有限K30の測定量に限定し、M05 C/Tは更新しない。主外側22.86秒／RSS231,129,088 bytes。全集計と指定18系列の生成内容の独立復元で不一致0、他35,982系列の完全生成監査ではない。

保存結果ビューを追加、全1,001 pytest成功（27.28秒）／新AppTest11／pip check成功。原標本保持・全12CSV値・4 payloadはAppTestで確認。実ブラウザ／実ダウンロード／HTML実表示はWork設定で利用不可のため未検証。HTTP200・healthと限定起動ログを確認し、確認用サーバーは停止済み。詳細・全条件・再現手順・残課題は `evidence/rq_m06/REPORT.md`／`validation.json`。新成果はレビュー待ち、044〜047 In Progress、Next044のみを維持し、Doneを代行しない。通常pytest／UI／監査から主実験を起動せず、追加実験しない。

再開入口は `mathscape/matched_stability.py`／`matched_stability_ui.py`、`scripts/run_matched_stability_guarded.py`／`run_matched_stability_benchmark.py`／`audit_matched_stability.py`。次の単一RQ候補は固定原標本・固定Bでの有限KがAse測定精度とsqrtの偏りに与える影響。別の事前固定が必要な提案のみで、未実行のまま停止する。

提出先は [stacked PR #6](https://github.com/Tone75/mathscape/pull/6)、科学成果SHA `64132bfd2a7a01ea592b4ebfc38e0fbabac73c53`、事前固定SHA `3af12e48bec70381a4fc8d70b9b25c8b1e95c405`。baseは受入済み `feat/rq-m05-matched-moments`。この参照追記は科学成果固定後の文書commitで、最新提出HEADは `gh pr view 6 --repo Tone75/mathscape --json headRefOid` で取得できる。Notion044〜047・Home・固定仕様へ結果／SHA／PR／検証範囲／資源／未解決事項を同期済み、PENDING_NOTION_SYNCなし。新成果はレビュー待ちで044〜047 In Progress、Next044のみを維持する。既存PR／main／タグ／Releaseは保持し、保護対象10 remote refsは開始時と一致。新タグ・Release・mergeなし。

## 最新の受入と実行状態：RQ-M05

Owner明示承認と最新Notionに従い、RQ-M04（提出HEAD `fb2216d6fc506dd74779104edad024c61bc9e1e1`、科学成果 `137efab0dc949359a58fc481676d6da6341b0408`、事前固定 `60101c72e6073c7350c9f21900b98afc19cbab00`、PR #4）は受入済み、MATH-036〜039 Done、H1／H2保留を保持する。下記の旧レビュー待ちは提出時点の履歴であり開始条件ではない。merge・外部科学レビュー・Owner本人の個別新操作とは区別する。

CODEX_RECEIPT RQ-M05-v1：開始HEADは承認済みと一致、working tree clean／別worktreeなし。新branch `feat/rq-m05-matched-moments`。固定仕様全文と末尾プロンプト・親子票・Homeを読んだ。既存732 pytest成功（15.42秒）。NextはMATH-040だけ、開始した040／041をIn Progressにして041→042→043の順に実行する。正本と受領は `evidence/rq_m05/SPEC.md`／`CODEX_RECEIPT.md`。

母平均0／分散1／歪度2のガンマと対数正規、n20／α.05／K1／B4999、seed0／1／2、R10000を固定。K1の端点SD／S／Ase／ApopはNA。主実験1回、指定別namespaceスモーク1回、事前固定・入力照合後に実行する。通常pytest・保存監査・UIから主実験を起動しない。旧実験証拠・既存PR／main／タグ／Releaseは保持し、新成果は別stacked PRのレビュー待ちとして提出する。

事前固定 `a8e87b3deb1a5256a200b28ea51301dec8aecb7e`、manifest SHA256 `d02a4b1d2ea1f26d96549e35014903c0c86df32dc8b8507bf81cc2f6dfdc8454`。指定smoke1回後、主1回で6セル・60,000原標本完了、再実行0／無効・欠測0。H1／H2とも実用差仮説不支持かつ、この固定条件・両指標の±1pp内の実用的同等性支持。ΔC99%区間[−0.796871,+0.390204]pp、ΔT[−0.420947,+0.814281]pp。BCa被覆G91.3767%／L91.1733%で、名目95%達成や普遍的な前三モーメント十分性へ拡張しない。

主CLI外側36.66秒、内部36.562958秒、Peak RSS230,817,792 bytes。独立実装が保存原標本・端点から全180,000レコード／18 seed／6 pooled／12分布差／24方法対応差と両仮説・全seedを再計算し不一致0。実bootstrap過程の完全復元監査ではない。数理・runner・監査・guard等16固定sourceは不変。

保存専用ビューを実装し全862 pytest成功（20.64秒、旧732＋新130）・pip check成功・新AppTest11件。実ブラウザで両分布／全方法／seed・ID／Lab・モード保持と復元、固定α／B／K、NA、21状態147表示数値、3種類の実ダウンロード一致、旧4 Labと旧保存ビュー代表操作を確認した。確認用サーバー停止済み。参照は `evidence/rq_m05/REPORT.md`／`validation.json`／`browser/`。

再開入口は `mathscape/matched_moments.py`／`matched_ui.py`、`scripts/run_matched_guarded.py`／`run_matched_benchmark.py`／`audit_matched.py`。新成果はレビュー待ちで040〜043 In Progress、Nextは040だけ。新成果のOwner受入・外部科学レビュー・Owner本人の個別新操作・別ブラウザ／モバイルは未実施。既存別改善票・MATH-005本格再開試験へ完了範囲を広げない。

提出先 [stacked PR #5](https://github.com/Tone75/mathscape/pull/5)、科学成果 `254af9ed19cf3197310bc372c704ecc32c3b9b4e`。baseはPR #4の承認済み `feat/rq-m04-distribution-shape`。この参照追記は科学成果固定後の文書commitで、最新提出HEADは `gh pr view 5 --repo Tone75/mathscape --json headRefOid` で取得できる。Notion040〜043・Home・固定仕様へ固定SHA／結果／検証／資源／PR／未解決事項を同期済み、PENDING_NOTION_SYNCなし。保守的な全変更公開成果約24.2MiB、50MiB内。保護対象remote 9 refs（main・旧4作業branch・既存2タグと参照先）は開始時と一致した。新タグ・Release・mergeなし、レビュー待ちとして提出し停止する。

次の単一RQ候補は「同じ前三母モーメントを持つG／Lの固定原標本で、BCa端点の条件付き再標本化変動はどの程度異なるか」。K1で推定不能だった量を別の事前固定で扱う提案のみで、追加実験は行わない。

## 最新の受入と実行状態：RQ-M04

2026-11-08のOwner明示承認とNotionの最新記録に従い、RQ-M03（提出HEAD `9b4a0fa1b3bea7751069deaaf0f3669125930403`、科学成果 `568c689a71d67bca393fa5000d7b855e7c609e45`、PR #3、H1／H2／H3支持）は受入済み、MATH-032〜035はDone。下記の旧受入待ちは提出時点の履歴であり開始条件ではない。外部科学レビュー・Owner本人の新機能個別操作・PR mergeとは区別する。

CODEX_RECEIPT RQ-M04-v1: 開始HEADは承認済みと一致、未コミット変更・別worktreeなし。新branch `feat/rq-m04-distribution-shape`。固定仕様全文・末尾プロンプト・親子票を受領し、037→038→039の順で実行する。NextはMATH-036だけ、開始した036／037だけIn Progress。既存612テスト成功（10.91秒）。詳細は `evidence/rq_m04/CODEX_RECEIPT.md`。新成果は別stacked PRでレビュー待ちとして提出し、旧成果／既存PR／main／タグ／Releaseは保持する。

事前固定 `60101c72e6073c7350c9f21900b98afc19cbab00`、manifest SHA256 `94dbdbfbe20acef3e1f19bcd847f6a2ced6792fc3588994f2f14a64d4f0dc3c2`。新規106参照テストと指定別namespaceスモーク1回後、guarded主実験1回で18セル・7,200原標本・72,000系列完了。再実行0・無効0、H1／H2保留。H1は99%区間が−2ppをまたぎseed0方向も未達、H2は99%区間が+3ppをまたぐ。保留を差ゼロ等と扱わず、追加試行をしない。

外側CLI44.30秒、worker内部44.14562025秒、Peak RSS253,296,640 bytes。保存原標本・端点から独立実装で43,200原標本別行、108 seed／36 pooled、72 B対応差／144手法対応差、全指標と2仮説・seed条件を再計算し不一致0。生bootstrap列は保存せず、実prefix生成の保証は固定ソース・参照テストと分けて説明する。

MATH-039の保存専用ビューを追加し、旧612を含む全732 pytest成功（16.01秒）、pip check成功、新AppTest14件。実ブラウザで正規／r=.5／r=1／r=16のB／k／手法／Lab・モード保持、分布変更時の別原標本と復元、固定α、22条件176表示値とCSV／NPZ、CSV・HTML実ダウンロード一致、旧4 Lab代表操作を確認。ログの例外・対象重複初期値警告なし、確認用サーバー停止済み。証拠は `validation.json`／`browser/`。新成果はレビュー待ちで親子票036〜039はIn Progress、Nextは036だけを維持する。新成果のOwner受入・外部科学レビュー・Owner本人の個別新操作・別ブラウザ／モバイルを実施済みにしない。

再開入口は `mathscape/distribution_shape.py`、`scripts/run_shape_guarded.py`、`scripts/run_shape_benchmark.py`、`scripts/audit_shape.py` と `evidence/rq_m04/`。主実験を再抽選せず、通常pytest・保存監査・UIから主実験を起動しない。RQ-M03以前の証拠・既存PR／main／タグ／Release・別改善票・外部科学レビューを保持する。

提出先 [stacked PR #4](https://github.com/Tone75/mathscape/pull/4)、科学成果 `137efab0dc949359a58fc481676d6da6341b0408`。baseはPR #3の承認済み `feat/rq-m03-stability`。この成果参照追記は科学成果固定後の文書commitで、最新提出HEADは `gh pr view 4 --repo Tone75/mathscape --json headRefOid` で取得できる。Notion036〜039・Home・固定仕様へ固定SHA／結果／検証／資源／PR／未解決事項を同期済み、PENDING_NOTION_SYNCなし。全変更公開成果約27.2MiB、50MiB内。保護対象remote 8 refs（main・旧3作業branch・既存2タグと各参照先）は開始時と一致。RQ-M04新成果の受入を代行せず、レビュー待ちで停止する。

次の単一RQ候補は「母平均・母分散・歪度を揃えた異なる分布族で、BCa被覆と非被覆方向はどう異なるか」。別の事前固定を要する提案のみで追加実行しない。既存別改善票・MATH-005本格再開試験・外部科学レビュー等の状態を今回の完了へ含めない。

## 最新の受入と実行状態：RQ-M03

2026-11-08のOwner明示承認に従い、RQ-M02（提出HEAD `dd3b377f02c19381b9fa6ffbcd0eff635911940c`、科学成果 `6a651c0995cbdd7a2de77d903c09cb23dbe369a8`、PR #2、H1支持・H2保留・H3支持）は受入済みとして扱う。下記の旧「受入待ち」は提出時点の履歴であり開始条件ではない。外部科学レビュー・Owner本人の個別新操作の実施とは区別する。

CODEX_RECEIPT RQ-M03-v1: 開始HEADは承認済み提出HEADと一致、未コミット変更・別worktreeなし。新branch `feat/rq-m03-stability`。固定仕様全文と末尾プロンプトを受領し、MATH-033→034→035を順に実施する。NextはMATH-032だけ。詳細は `evidence/rq_m03/CODEX_RECEIPT.md`。既存483テスト成功。主実験は事前固定commit後に1回のみ。新成果はレビュー待ちで提出する。

事前固定commit `e12faa6c34917fec42856e04cd806a7d5ac6a2a8`、manifest SHA256 `aec36ae1d78cd406d0fa6503c832118a7550e15bb3f5842000343c918c526534`。指定別namespaceスモーク1回後、主実験1回で12セル・1,200原標本・24,000系列を完了。再実行0、無効0、H1／H2／H3支持。H2は関連、H3は有限B=4,999の結果に限る。最大Bは真値ではない。全数値は `evidence/rq_m03/REPORT.md`／CSV／`hypotheses.json` を参照。

主CLI外側wall13.62秒、内部13.563085292秒、Peak RSS225,509,376 bytes。別実装監査が保存原標本・端点から10800原標本別行、108 seed集計、36 pooled、8 H2群、48被覆差、4確認量を再計算し不一致0。全bootstrap列は保存していないため実prefix生成は固定ソース・参照テストで検証し、保存データ監査の範囲と区別する。外部科学レビューは未実施。

再開の入口は `mathscape/stability.py`、`mathscape/stability_ui.py`、`scripts/run_stability_benchmark.py`、`scripts/audit_stability.py` と `evidence/rq_m03/`。主実験は再抽選せず、保存結果を利用する。通常pytest・UI・監査は主実験を起動しない。保存専用ビューを実装し、全612 pytest成功（10.95秒、旧483＋新129）・pip check成功。AppTest14件、実ブラウザで3条件・全B・k／手法／Lab／モード保持・固定α・CSV実ダウンロード一致・旧4 Lab代表操作を確認。画面で発見した原標本マーカーの非表示を修正し再検証した。証拠は `validation.json`／`browser/`。確認用サーバー停止済み。新PR提出はレビュー待ちとし、main・旧PR／タグ／Release、旧RQ-M01／M02証拠、別改善票は保持する。

提出先 [stacked PR #3](https://github.com/Tone75/mathscape/pull/3)、科学成果 `568c689a71d67bca393fa5000d7b855e7c609e45`。この参照は科学成果固定後の文書commitで追記した。最新提出HEADは `gh pr view 3 --repo Tone75/mathscape --json headRefOid` で確認できる。Notion親子票・Home・固定仕様への同期済み、PENDING_NOTION_SYNCなし。全変更成果約12.4MiB（保守的な全変更ファイル合計、50MiB内）。新成果はレビュー待ちで親子票はIn Progress、NextはMATH-032だけ。外部科学レビュー・Ownerの個別新操作／今回受入・別ブラウザ／モバイルは未実施。次の単一RQは固定n／分散の標準化ガンマで歪度を変える候補のみを提案し、追加実行せず停止する。

## 最新の受入と実行状態：RQ-M02

2026-11-08のOwner明示指示とNotionのOWNER_ACCEPTANCEを採用する。RQ-M01 / v0.2（科学成果 `cc240b0e32b424bb8881c153f17e56608d5fe51f`、提出HEAD `1dd298db671ad962fb705b5f407a9c1f51a11bb9`）はOwner受入・ユーザー動作確認ともに完了済み。下記の旧「Owner待ち」は提出時点の履歴であり、再確認を開始条件にしない。外部科学レビューや別改善票の完了を意味しない。

CODEX_RECEIPT RQ-M02-v1: 固定仕様全文と末尾プロンプトを受領。開始HEADは承認済み提出HEADと一致し、未コミット変更・別worktreeなし。ローカルAGENTS.mdなし。新ブランチ `feat/rq-m02-bootstrap` でMATH-029→030→031を進める。NextはMATH-028のみ。PR #1未mergeのため、承認済み `feat/inference-lab-v0.2` をbaseにしたstacked PRで新差分を分離する。main・旧PR・タグ・Release・RQ-M01証拠は変更しない。

新成果は `evidence/rq_m02/`。数理・UI・固定実験・検証・科学レポートを実装済み、新成果はレビュー待ちでありOwner受入を代行しない。旧RQ-M01の固定実験は再実行せず、ソースと証拠を保全した。

事前固定commit `2496eb06d77593e8ea2f19834de3b7aed2675a07`、manifest SHA256 `cd73e65b711422fc1a1d17052a0ebfa0b1639869ab5839d612bd3a2171049ec8`。別namespaceのスモーク1回36標本組（R2/B99）後、主実験1回を実施し再実行0。18生成セル・36,000標本組・54手法行、無効0。H1支持、H2保留、H3支持。H2は改善差の99%区間が事前閾値1ポイントをまたぐ。

全pytest483件成功（既存362＋数理78＋実行器27＋UI16）、実ブラウザで3条件・α/方法/モード/Labの保持、未適用表示、保存CSVとの108セル照合、旧4 Lab代表操作を確認した。独立監査は保存端点からの再計算で、追加の原標本・再標本生成なし。主CLI外側wall14.62秒、Peak RSS258,998,272 bytes。詳しい結果・全条件表・再現手順・停止規約は `evidence/rq_m02/REPORT.md`、操作証拠は `browser/validation.json`。

再開時は `mathscape/bootstrap.py`、`mathscape/bootstrap_ui.py`、`scripts/run_bootstrap_benchmark.py` と対応テストを入口とする。通常の `python -m pytest -q` は主実験を起動しない。主実験を再実行する必要はなく、不支持・保留を理由に追加実験しない。コード不具合修正時の最大1回再実行も旧結果・理由・修正版を残す別attemptと累計wall≤600秒が必要。

提出先は [stacked PR #2](https://github.com/Tone75/mathscape/pull/2)、実装・科学成果commitは `6a651c0995cbdd7a2de77d903c09cb23dbe369a8`。baseは `feat/inference-lab-v0.2`。最新の提出HEADは `gh pr view 2 --repo Tone75/mathscape --json headRefOid` で確認できる。この提出参照は科学成果固定後の文書commitで追記している。

Notion親子票・Home・固定仕様への成果参照に、受領・固定成果SHA・PR・483テスト・主実験1回・資源・画面検証・新成果レビュー待ちを同期済み。PENDING_NOTION_SYNCなし。次の単一RQは有限B_bootによるBCa尾部端点の安定性を候補として提案しただけで、今回は実行していない。旧改善票MATH-006/011/027、MATH-005本格再開試験、外部科学レビューは未完了のまま保持する。

## 以下はRQ-M01提出時点の履歴

CODEX_RECEIPT: 2026-11-08。MATH-021を受領し、MATH-024→025→026の実装・固定検証・成果提出を行う。基準はv0.1.1 / bc3c6adb963ecd501cf3c53bd647dfb42dd8d8ec。開始時のローカル・origin/mainは一致し、未コミット変更なし、既存216テスト成功。ローカルAGENTS.mdは存在せず会話の指示を適用。

ブランチ: `feat/inference-lab-v0.2`。親票MATH-021だけがNext。新タグ・Release・mainへのマージは今回行わない。Owner受入、MATH-002の利用評価、MATH-005の本格的な別セッション再開試験、MATH-018全体は未実施であり開始条件ではない。

## 再開の入口

- 数理: `mathscape/inference.py` と `tests/test_inference.py`
- UI: `mathscape/inference_ui.py` と `tests/test_inference_app.py`
- 固定設計・結果・科学的な限界: `evidence/rq_m01/manifest.json`, `summary.csv`, `REPORT.md`
- 実行器: `scripts/run_inference_benchmark.py`。固定実験は再抽選しない。既存成果を上書きせず、再実行の必要性・制約はREPORTを先に確認する。
- 通常検証: `python -m pytest -q`。アプリ: `python -m streamlit run app.py`。

Notion同期・固定成果・検証結果は以下へ記録する。成果提出はOwner受入ではない。

## 提出記録

数理・UI・54ケース固定実験・レポートを実装済み。既存216件を含むpytest362件成功（7.14秒）、新LabのAppTest16件、実ブラウザで3プリセット・α変更時とLab往復の標本保持・旧3 Lab代表操作を確認。固定実験は1回だけで再実行0、無効0、全54セル完了。H1〜H3は事前基準の限定範囲で支持。

事前固定ソース: `f249efec49478b3d43fb68f8cb8d43b9f2ead58d`。manifest SHA256: `4d9fd6044277ebe6b2ba06c028869cfa680fa8c26d424eb9697a31d2aadd8348`。最終成果はこの文書を含む作業ブランチのcommitとPRで固定する。自己参照を避け、最終HEADのSHAはNotion親票・HomeとPRへ記録する。

CLI全体ツール観測wall0.602秒、実験内部wall0.183秒、Peak RSS103,809,024 bytes。新しい依存関係・課金・Webホスティング・タグ・Release・mainマージはなし。固定実験を繰り返す必要はない。結果を望ましい形にする再抽選はしない。

提出先は [PR #1](https://github.com/Tone75/mathscape/pull/1)。実装・科学成果を固定したcommitは `cc240b0e32b424bb8881c153f17e56608d5fe51f`（この提出参照の追記はその後の文書commit）。PRの最新HEADは `gh pr view 1 --repo Tone75/mathscape --json headRefOid` で確認できる。

Notion同期は完了し、親票・MATH-024〜026・Project Home・固定仕様への成果参照に、受領・固定SHA・PR・検証・レビュー待ちを反映した。PENDING_NOTION_SYNCなし。親票と子票はOwnerレビュー待ちとしてIn Progressを維持し、Doneにしていない。

未解決: Owner受入・利用評価・独立した科学レビュー、MATH-005の別セッション再開試験は未実施。次の候補RQはREPORTに1件だけ記載し、実行は停止する。続きの判断前にPR・Notionの最新状態とローカル差分を確認する。
