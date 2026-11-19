# CODEX_RECEIPT · RQ-M11-v1

2026-11-19 JST。MATH-064を受領。開始HEADは `bd01ae7ff55b1e8593a4b3bd003587db2732c050`。PR #10はOpen、同一HEADの作業ツリーはcleanでworktreeは1個。ローカル追加AGENTS.mdはなく、対話のグローバル指示を適用。README、HANDOFF、Notion固定仕様・064〜067・Homeと受入記録を確認した。

M10のH1/H2はG/Lとも保留としてOwner受入済み。M08/M09の限定的支持を保持する。保存M10の結果を知った後の決定的な追加解析として、M11参照・誤差を実データで算出する前に定義・ソース・入力を固定する。主解析と監査の起動担当は1名。新しい原標本、乱数、smokeは使わない。Nextは064のみ。提出先はM10をbaseとする別PR。

MATH-064/065のStatusをIn Progressへ更新し、依存先の066/067は開始前のまま。解析担当・独立監査担当・保存ビュー契約担当に分担した。入力の照合には受入commitのGit blobとSHA-256を使う。主解析・独立監査の起動は同一担当が順番に行う。

- 仕様 [RQ-M11-v1](https://app.notion.com/p/415251c5f085815e8d5ee71a429b3a48)
- 親票 [MATH-064](https://app.notion.com/p/415251c5f085816f9a38f27f0a78f5d8)
- 作業ブランチ `feat/rq-m11-enumerated-reference`、base `feat/rq-m10-small-sample`

M10で新たに生成・保存された原標本と、その有限Bの数値規約を照合する。M11で新たに行う比較はM10のBCa／percentileの同じ系列を1回ずつ使う。M10のt区間と3配分の集計は受入記録を保持する。通常テストは固定fixtureと既存アルゴリズムの回帰で、実データの主解析は専用ランチャーからのみ実施する。
