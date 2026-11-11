# CODEX_RECEIPT — RQ-M10-v1

受領日：2026-11-11（Asia/Tokyo）。正本の固定仕様全文・末尾「Codex実行プロンプト｜RQ-M10-v1」、MATH-060〜063、Project Homeを読んだ。仕様は `SPEC.md` に公開可能な形で保存し、個人ローカル絶対パスだけを置換した。

- 開始HEAD：`1bf925b55a838650ee020fe09f806cf64e8ebc02`（Owner受入済みM09提出HEADと一致）。
- M09科学成果：`de49a3a59aebb474ede35b320e82956ba45cc1dc`、事前固定：`9f3e5d6245e5c22a636795baaf9eb8297452d8ba`。PR #9はOpen／未merge／HEAD一致を確認。M09 H1/H2 n20両分布支持、056〜059 Done／Next=falseを保持する。
- 開始tree clean、別worktreeなし。originは `https://github.com/Tone75/mathscape.git`。新branch `feat/rq-m10-small-sample` を受入HEADから作成。既存branch/main/PR/タグ/Releaseは変更しない。
- cwd実体・権限を確認。ルート〜projectのAGENTS.mdおよびproject配下のAGENTS.mdは見つからなかった。ユーザー提示のグローバルAGENTS指示（チーム分担・生成ファイルのフルパス出力）を適用。README/HANDOFFを読み、旧レビュー待ちは当時の履歴とし再承認を開始条件にしない。
- アプリの並行チャット一覧とworktreeを確認。同じMATHSCAPEの別作業は見つからず、EVOLAB／CIVITAS／他PJのチャット・ファイルへ変更しない。数理／runner／独立監査を分担し、主と指定smokeの起動担当はrootの1名。
- 060受領→061生成前固定→062主1回／独立監査→063保存UI／検証／提出／同期。開始した060/061だけIn Progress、Next060のみ。新成果Done・Owner受入を代行しない。
- 新60原標本n5、主namespace `0x4D3130`、smoke `0x534D3130`。原標本6要素・内側7要素の固定キー。M10主原標本はまだ生成・閲覧していない。存在しない原標本hashを事前固定済みとは記載しない。
- G/L各N30・B4999・R20・共通120系列。BCa P120対P30のd/L/U99.5%近似t区間6本、H1±.0025・H2(.8,1.25)。原標本t df4／比較df29、SE=s/sqrt5、Apop=1/sqrt5。旧新pool・旧新ID対応・n20対n5差検定なし。
- 指定smokeはM09保存6原標本の先頭5値だけ、元／投影hash保存、R2/B99/共通12系列を1回。追加ランダムpilotなし。主は全raw/x保存・読み戻し後に内側144000系列へ進む。
- 主1回、外側300秒／Peak RSS2GiB／公開追加50MiB／batch100万要素以下。結果・警告・無効・資源超過を理由とする再抽選・増量・自動縮小なし。明確な実コード不具合だけ旧証拠保全・同じID/key/生成済み標本reuseで別attempt最大1回、主累計600秒、attemptを合算しない。
- 数理的無効と警告を分離し、無効を0/非被覆へ置換しない。必要BCa無効は原標本主量NA、Nを縮めず保留。conditional_は参考のみ。全列挙は実装しない。
- 外部科学レビュー・Owner個別新操作・mergeは受入や技術検証と区別。旧未検証範囲を保持。依存追加/更新・課金・GPU・ホスティング・自動化・他PJ変更なし。

受領時点では固定実験・新科学結果・最終検証は未実施。以後の事前固定・実行・検証・提出記録はmanifest／resources／REPORT／validationへ実測とSHAを追記する。

生成前の基準検証：旧tracked tests全1,396件成功（94.08秒）、pip check成功。環境はmacOS26.6.2／arm64／Python3.12.13、NumPy2.5.4／SciPy1.18.1／Streamlit1.67.0／Plotly6.9.0／pytest9.2.0／pandas3.1.1。依存追加・更新なし。
