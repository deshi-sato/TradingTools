# scripts/stream_microbatch_rest.py 仕様書

## 概要
- PUSHで受信したティックとRESTで取得した板情報を組み合わせ、`tick_batch` / `orderbook_snapshot` / `features_stream` を更新するマイクロバッチ処理。
- WebSocketが利用できない環境でも板情報をRESTポーリングで補完できるよう設計されている。

## 主な機能
- 1シンボルあたり `window_ms` 間隔でティックを集計し、uptick/downtick・出来高合計・最終価格を算出。
- `BoardFetcher`（REST版）で `/kabusapi/board/<symbol>@1` をポーリングし、取得できない場合は安全なデフォルトを適用。
- `feature_calc` のユーティリティを用いて `spread_bp`、`top3_sum`、`depth_imbalance`、`uptick_ratio` を計算し、`features_stream` に書き込む。
- `mock` モードでは内蔵のモックティック発生器がランダム価格を生成し、オフライン環境で挙動検証できる。

## 設定項目（config/stream_settings.json）
| キー | 説明 |
|------|------|
| `mode` | `online` / `mock` の既定値。CLIで上書き可能。|
| `host`, `port` | kabuステーションAPIのホスト・ポート。RESTとWebSocketで共有。|
| `token` | WebSocket/PUSHのX-API-KEY。トークン未設定時はRESTのみ利用可能。|
| `symbols` | 処理対象シンボル配列。CLI `--symbols` で上書き可。|
| `window_ms` | マイクロバッチの長さ（ミリ秒）。|
| `tick_queue_max` | ティックキューの最大保持数。|
| `mock_interval_ms` | `mock` モードでティックを生成する間隔。|
| `price_guard` | 銘柄ごとの価格上下限設定。|
| `rest_poll_ms` | 板取得の最小ポーリング間隔（BoardFetcher）。|
| `db_path` | SQLite DBパス。|

## CLI引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。設定JSON。|
| `-Verbose` | ログレベル（1=INFO, 0=WARNING）。|
| `--mode` | `online` / `mock` を明示指定。|
| `--symbols` | カンマ区切りで対象シンボルを指定。|
| `--probe-board` | 起動時に `/board` を1回呼び出して疎通チェック。

## RESTボード取得
- `BoardFetcher` は `rest_poll_ms` で間引きしながら `/kabusapi/board` をポーリング。
- 取得成功時は `bid1` / `ask1` / 上位3本の板数量を返却し、失敗時は空の板を返して安全にスキップ。

## 処理フロー
1. 設定と引数を読み込み、対象シンボル・動作モード・ログレベルを確定。
2. SQLiteで `tick_batch` / `orderbook_snapshot` / `features_stream` テーブルとインデックスを作成。
3. `online` モードではWebSocket受信スレッドを立ち上げ、`mock` モードでは擬似ティック生成スレッドを起動。
4. 各バッチ窓でティックを集約し、板情報をRESTから取得して`tick_batch`/`orderbook_snapshot`/`features_stream`へINSERT。
5. バッチごとに処理件数をINFOログ出力し、KeyboardInterruptで停止要求を受けたらスレッド終了とDBクローズを行う。

## ログ
- `[BOOT]` や `batch ticks=...` などのINFOログで進行状況を報告。
- REST取得失敗時や価格ガード違反はWARNINGログとして出力。

## 実行例
```powershell
py scripts/stream_microbatch_rest.py -Config config/stream_settings.json --mode online --probe-board
py scripts/stream_microbatch_rest.py -Config config/stream_settings.json --mode mock -Verbose 1
```
