# scripts/ws_ingest_min.py 仕様書

## 概要
- kabuステーションのPUSH WebSocketに接続し、指定秒数だけ受信したら簡易テーブル `orderbook_snapshot` にタイムスタンプを蓄積する疎通テストスクリプト。
- 最低限のテーブル作成とコミットのみ実装されており、WebSocketの安定性や接続維持時間を確認する目的で使用する。

## 主な機能
- 設定JSON (`-Config`) から `port` / `token` / `db_path` を読み込み、WebSocket接続を確立。
- 受信したメッセージの内容は破棄し、`orderbook_snapshot(ts)` テーブルに現在時刻を挿入。
- 一定件数ごとにコミットしながら進捗ログを出力し、例外発生時は途中でループを抜ける。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`port`/`token`/`db_path` を含むJSON。|
| `-Seconds` | WebSocket受信を続ける秒数。既定600。|

## 処理フロー
1. 設定を読み込み、WebSocketで `ws://localhost:<port>/kabusapi/websocket` へ接続。
2. `orderbook_snapshot` テーブルが存在しなければ作成。
3. 受信ループでメッセージを読み捨て、JST現在時刻を `ts` 列としてINSERT。20件ごとにコミットして進捗表示。
4. タイムアウト・例外時にはログを出力してループを終了し、最後に接続とDBをクローズ。

## 入出力
- 入力: WebSocketメッセージ（内容は保存しない）。
- 出力: SQLite `orderbook_snapshot` に `ts` 列のみの行を追加。
- ログ: `[INGEST] inserted=...` / `[ERR] ...` を標準出力へ表示。

## 実行例
```powershell
py scripts/ws_ingest_min.py -Config config/stream_settings.json -Seconds 120
```
