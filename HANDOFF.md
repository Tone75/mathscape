# MATHSCAPE — minimal handoff

## 最新の受入と実行状態：RQ-M02

2026-11-08のOwner明示指示とNotionのOWNER_ACCEPTANCEを採用する。RQ-M01 / v0.2（科学成果 `cc240b0e32b424bb8881c153f17e56608d5fe51f`、提出HEAD `1dd298db671ad962fb705b5f407a9c1f51a11bb9`）はOwner受入・ユーザー動作確認ともに完了済み。下記の旧「Owner待ち」は提出時点の履歴であり、再確認を開始条件にしない。外部科学レビューや別改善票の完了を意味しない。

CODEX_RECEIPT RQ-M02-v1: 固定仕様全文と末尾プロンプトを受領。開始HEADは承認済み提出HEADと一致し、未コミット変更・別worktreeなし。ローカルAGENTS.mdなし。新ブランチ `feat/rq-m02-bootstrap` でMATH-029→030→031を進める。NextはMATH-028のみ。PR #1未mergeのため、承認済み `feat/inference-lab-v0.2` をbaseにしたstacked PRで新差分を分離する。main・旧PR・タグ・Release・RQ-M01証拠は変更しない。

新成果は `evidence/rq_m02/`。数理・UI・固定実験・検証・科学レポートを実装済み、新成果はレビュー待ちでありOwner受入を代行しない。旧RQ-M01の固定実験は再実行せず、ソースと証拠を保全した。

事前固定commit `2496eb06d77593e8ea2f19834de3b7aed2675a07`、manifest SHA256 `cd73e65b711422fc1a1d17052a0ebfa0b1639869ab5839d612bd3a2171049ec8`。別namespaceのスモーク1回36標本組（R2/B99）後、主実験1回を実施し再実行0。18生成セル・36,000標本組・54手法行、無効0。H1支持、H2保留、H3支持。H2は改善差の99%区間が事前閾値1ポイントをまたぐ。

全pytest483件成功（既存362＋数理78＋実行器27＋UI16）、実ブラウザで3条件・α/方法/モード/Labの保持、未適用表示、保存CSVとの108セル照合、旧4 Lab代表操作を確認した。独立監査は保存端点からの再計算で、追加の原標本・再標本生成なし。主CLI外側wall14.62秒、Peak RSS258,998,272 bytes。詳しい結果・全条件表・再現手順・停止規約は `evidence/rq_m02/REPORT.md`、操作証拠は `browser/validation.json`。

再開時は `mathscape/bootstrap.py`、`mathscape/bootstrap_ui.py`、`scripts/run_bootstrap_benchmark.py` と対応テストを入口とする。通常の `python -m pytest -q` は主実験を起動しない。主実験を再実行する必要はなく、不支持・保留を理由に追加実験しない。コード不具合修正時の最大1回再実行も旧結果・理由・修正版を残す別attemptと累計wall≤600秒が必要。

Notion親子票・Homeへ受領・着手・事前固定を反映済み。最終成果SHA・PR・レビュー待ちの同期は提出時に記録する。次の単一RQは有限B_bootによるBCa尾部端点の安定性を候補として提案しただけで、今回は実行していない。旧改善票MATH-006/011/027、MATH-005本格再開試験、外部科学レビューは未完了のまま保持する。

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
