# scripts/naut_runner.py 仕様書

## 概要
- `features_stream` に蓄積されたティック特徴量を監視し、売買方向ごとのシグナルを判定して Naut 実行系へ通知するメインループ。
- プロファイル（`prod` / `prodlite` / `demo`）ごとに閾値セットを切り替え、バースト指標やストリーク整合性を考慮した採用制御を行う。

## 主な機能
- `compute_base_score()` で uptick 比率・板アンバランス・出来高超過から0〜1の基礎スコアを計算。
- `burst_helper.burst_bonus()` を適用して直近のバースト勢いを指数減衰で加点し、クールダウン秒数を超えたもののみ採用。
- `PollTracker`（内部実装）でシンボルごとの方向ストリークを管理し、一定回数連続で条件を満たした場合だけ以降の判定へ進む。
- 閾値や板厚・スプレッド条件を `settings.naut` からロードし、プロファイルごとに `BUY` / `SELL` の要件を細かく調整。
- `--print-summary` 指定時には1ループのみ実行し、棄却理由や採用件数の内訳を標準出力へダンプ。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`settings.naut` とDBパスを含むJSON。|
| `-Verbose` | ログレベル（1=INFO、0=WARNING）。|
| `--profile` | `prod` / `prodlite` / `demo` から閾値セットを選択。|
| `--print-summary` | 1サイクルのみ走らせ、採用/棄却統計を表示して終了。|

## 設定項目（例）
| キー | 説明 |
|------|------|
| `settings.naut.db_path` | `features_stream` を保持するSQLiteのパス。|
| `settings.naut.profile.PROFILE.BUY/SELL` | uptick閾値、アンバランス要求、スプレッド上限、ボリューム閾値などを定義。|
| `settings.naut.window` | ストリーク判定に使用する窓幅。|
| `settings.naut.cooldown_sec` | シンボルごとの再発火までの待機秒。|
| `settings.naut.market_window` | `HH:MM-HH:MM` 形式で取引時間を限定。|

## 処理フロー
1. 設定を読み込み、指定プロファイルの BUY/SELL 閾値・ウィンドウ長・ボリューム最小値などを決定。
2. SQLiteから最新フィーチャーをバッチ取得し、シンボルごとに uptick 履歴と方向ストリークを更新。
3. BUY/SELL候補について、spread条件・アンバランス投票・ボリューム条件を満たすか検証。
4. 条件を満たした場合に `compute_base_score()` → `burst_bonus()` を適用し、閾値を超えたら採用。`print_summary` が有効なら統計へ記録。
5. 採用したシンボルはログに詳細（base/adj/bonus/thr等）を出力し、冷却タイマーと採用回数カウンタを更新。
6. `--print-summary` でない場合は 0.25 秒間隔でループし続け、Ctrl+C で停止可能。

## 生成ログ
- `[BUY]` / `[SELL]` 行で採用シグナルの近接情報を出力。
- `[DBG overlap ...]` などのデバッグログは `-Verbose` や設定値により制御。
- `print-summary` モードでは `DBG accepted` / `DBG skipped` などの統計を標準出力に表示。

## 連携
- `scripts/features_burst_worker.py` が付与した `burst_*` 列を参照してボーナス計算を行う。
- `scripts/replay_naut.py` はこのロジックをリプレイ用にラップしている。
- 採用結果はログ出力を通じて `orchestrate/run_intraday.py` や通知系へ取り込み可能。

## 実行例
```powershell
py scripts/naut_runner.py -Config config/stream_settings.json --profile prod
py scripts/naut_runner.py -Config config/stream_settings.json --profile demo --print-summary
```
