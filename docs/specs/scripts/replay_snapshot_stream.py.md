# scripts/replay_snapshot_stream.py 仕様書

## 概要
- 休場日など実データが止まっている状況で、`rss_snapshot.db` の蓄積済みティック/板情報を1件ずつ再生し、`features_stream` へ PUT するリプレイ用スクリプト。
- `tick_batch` → `orderbook_snapshot` の順で参照し、実運用時の `stream_microbatch` と同じ 14 カラム構造を `insert_features` で書き込む。
- `tick_batch` が空の場合は `today_data` から簡易ティックを生成してフォールバック。

## 主要入力
| 入力 | 説明 |
|------|------|
| `-Config` | `config/stream_settings.json` など既存設定ファイル。`snapshot_db`/`features_db`/`symbols` の既定値を引き継ぐ。|
| `--source-db` | 元データとなる `rss_snapshot.db` のパス。未指定時は `-Config` またはプロジェクトルートの DB を使用。|
| `--target-db` | `features_stream` を挿入する先。未指定時は `--source-db` と同じ。|
| `--date` | リプレイ対象営業日 (`YYYY-MM-DD`) 。未指定時は最新日付を自動検出。|
| `--symbols` | 対象銘柄 CSV。未指定時は設定ファイルの `symbols[]` を流用。|

## 再生オプション
| オプション | 機能 |
|-------------|------|
| `--speed` | 再生倍率 (1.0=実時間)。時間スケールを調整。|
| `--no-sleep` | タイミング待ちを無効化して一気に書き込む。|
| `--max-sleep` | 1バッチあたりの最大待機秒数 (デフォルト 2.5s)。|
| `--limit` | 再生件数の上限。0 で全件。|
| `--truncate` | 対象日の `features_stream` を事前に削除してから再生。|
| `--quiet` | コンソール出力を抑制。|

## naut_runner 連携
- `--run-naut` を付けると再生と同時に `py scripts/naut_runner.py -Config <path>` を起動する。
- 追加引数は `--naut-extra "--print-summary --profile demo"` のように渡せる。
- 再生終了後は `--naut-grace` 秒待機した上で `terminate()` → `kill()` を行う。`--leave-naut` 指定時は終了させない。

## 主な処理フロー
1. `ensure_tables(target_db)` で `tick_batch/orderbook_snapshot/features_stream` の存在を保証。
2. 日付／銘柄を決定し、`fetch_tick_batches()` でティックウィンドウを取得。
3. 行が無ければ `fetch_today_rows()` で 1 分足から最小限のティックを作成。
4. 各バッチに対して `lookup_orderbook()` で最新板を引き当て、`build_feature_row()` で 14 カラム辞書を組み立て。
5. `stream_batches()` が速度調整しつつ `insert_features()` を呼び出し、ログを標準出力へ送出。

## 利用例
py -m scripts.replay_snapshot_stream -Config config\stream_settings.json --truncate --speed 0 --run-naut --naut-extra="--print-summary"
