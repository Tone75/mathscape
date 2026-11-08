# MATHSCAPE v0.2 / RQ-M01 — minimal handoff

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

Notionは接続でき、CODEX_RECEIPT・着手・数理検証を反映済み。提出時に最終SHA・PR・検証・レビュー待ちを追記する。PENDING_NOTION_SYNCは現時点なし。親票と子票はOwnerレビュー待ちとしてIn Progressを維持し、Doneにしない。

未解決: Owner受入・利用評価・独立した科学レビュー、MATH-005の別セッション再開試験は未実施。次の候補RQはREPORTに1件だけ記載し、実行は停止する。続きの判断前にPR・Notionの最新状態とローカル差分を確認する。
