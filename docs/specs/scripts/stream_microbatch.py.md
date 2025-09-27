# scripts/stream_microbatch.py 仕様書

## 概要
- kabuステーション PUSH (WebSocket) から株価ティックを受信し、マイクロバッチで `tick_batch` と `features_stream` に書き込む常駐ワーカー。
- `mode=online` 時は WebSocket PUSH を購読、`mode=mock` 時はランダムティックを生成して挙動確認に使える。
- 既存の `BoardFetcher` と組み合わせ、ティックが無いタイミングでも REST 板を取得して `orderbook_snapshot` を埋める。

## 主な責務
- PUSH メッセージの正規化 (`Symbol`, `CurrentPrice`, `TradingVolume` 等) → バッチ投入。
- 出来高累積から差分を計算し、ティックごとの数量を洗い替えする。
- 価格レンジのガード (銘柄ごとの min/max) を通過したものだけキューに投入。
- バッチ確定後に `tick_batch`, `orderbook_snapshot`, `features_stream` へ書き込み。
- `--probe-board` 指定時は起動時に /board REST 疎通チェックを行う。

## 設定 (config/stream_settings.json)
| キー | 型 | 説明 |
|------|----|------|
| `mode` | `"online"` / `"mock"` | WebSocket を使うか、ダミー生成にするか。CLI `--mode` で上書き可能。|
| `host` / `port` | str / int | kabuステAPI の接続先。デフォルト `localhost:18080`。|
| `token` | str | `X-API-KEY` に使う API トークン。未指定時は環境変数 `KABU_TOKEN` / `KABU_API_KEY` を参照。|
| `price_guard` | dict | シンボルごとの価格範囲。`{"7203": {"min": 400, "max": 6000}}` のように指定。範囲外はログ警告のうえ破棄。|
| `websocket` | dict | PUSH 受信に使うタイムアウトやリトライ設定。`connect_timeout`, `recv_timeout`, `backoff_initial`, `backoff_max`, `guard_log_interval` を受け付ける。|
| `tick_queue_max` | int | PUSH から集約キューへのバッファ数。既定 20000。|
| `mock_interval_ms` | int | モックモード時のティック発生間隔。既定 50ms。|
| `window_ms` | int | バッチ幅 (ミリ秒)。既定 300ms。|
| `symbols` | list[str] | 監視銘柄。CLI `--symbols` で上書き可。|
| `db_path` | str | SQLite の保存先 (通常 `rss_snapshot.db`)。|
| `log_path` | str | ログファイル出力先。|
| `board_mode` / `rest_poll_ms` | str / int | `BoardFetcher` 用の設定。|

## コマンドライン引数
| 引数 | 説明 |
|------|------|
| `-Config <path>` | 必須。JSON 設定ファイルを指定。|
| `-Verbose` | ログレベル制御。1 で INFO、0 で WARNING。|
| `--mode` | 設定ファイルより優先して `online` / `mock` を切り替え。|
| `--symbols` | CSV 指定で監視銘柄を差し替え。|
| `--probe-board` | 起動時に最初の銘柄で `/board` REST 疎通確認を実施しログ出力。|

## 処理フロー
1. 設定ロード → `mode` 判定 → 銘柄リスト決定。
2. WebSocket 接続パラメータ・価格ガードを初期化。`mode=online` の場合、トークン必須。
3. ログ初期化 (`logging.basicConfig`) とブートメッセージ出力。必要に応じて `/board` 疎通チェック。
4. SQLite を WAL/NORMAL にセットし、`ensure_tables` でテーブル存在保証。
5. `mode=online` → `PushTickReceiver` を生成し WebSocket 接続。ネットワークエラー時は指数バックオフで再接続。`mode=mock` → `MockTickReceiver` がランダムティックを発生。
6. PUSH 受信時は JSON を正規化：シンボル判定、価格ガード、時刻整形、出来高差分を計算し `(symbol, price, size, ts)` をキューへ送る。
7. `window_ms` 間隔でキューを吸い上げ `ticks_buf` を銘柄ごとに蓄積。ティックが無い場合も REST 板を取得して `orderbook_snapshot` を更新。
8. ティックがある銘柄は uptick/downtick/出来高合計/最終価格を算出し `tick_batch` へ insert。板情報から features を組み立て `features_stream` へ書き込み。
9. ループ脱出時 (`KeyboardInterrupt` 等) にはスレッド停止・DB クローズ・ログ出力を行う。

## ログと監視
- `[BOOT] mode=...` で起動パラメータを標準出力に表示。
- `[GUARD] price guard configured ...` でガード設定の件数を INFO 出力。
- 価格ガード違反は WARN (`price guard drop ...`)。
- キューあふれは `[PushTickReceiver]` ログで WARN し、5秒単位で統計を表示。
- WebSocket 接続・切断・再接続は `PushTickReceiver` が INFO/WARN ログを出す。

## 外部スクリプトとの連携
- PUSH 受信前に `scripts/register_watchlist.py` で銘柄登録 (`PUT /kabusapi/register`) を済ませる運用。
- Offline 再生時は `scripts/replay_snapshot_stream.py` を使用し、本スクリプトは停止しておく。
- downstream の `scripts/naut_runner.py` は `features_stream` をポーリングしてシグナルを判断する。

## サンプル起動コマンド
```powershell
# Online モード (PUSH 実行)※事前に register_watchlist.py を実行しておく
py scripts/stream_microbatch.py -Config config/stream_settings.json --probe-board

# Mock モード（ダミーティック生成）
py scripts/stream_microbatch.py -Config config/stream_settings.json --mode mock -Verbose 1
```
