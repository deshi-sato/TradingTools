# scripts/ws_probe.py 仕様書

## 概要
- kabuステーションWebSocketの疎通確認用ツール。指定時間だけ受信し、メッセージ内容の冒頭を標準出力に表示する。
- PUSH配信の有無やトークン有効性を手軽に確認できる。

## 主な機能
- 設定JSONから `port` と `token` を読み取り、WebSocket接続を確立。
- 受信したメッセージを200文字程度にトリミングして `[MSG n]` 形式で表示。
- タイムアウト時は通知しつつループ継続、例外時はエラーメッセージを表示して終了。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`port` と `token` を含む設定ファイル。|
| `-Seconds` | 受信を継続する秒数。既定15。|
| `-Verbose` | 接続開始時のログを制御（1=表示, 0=静か）。|

## 処理フロー
1. 設定を読み込み、`ws://localhost:<port>/kabusapi/websocket` へ接続。
2. 指定秒数までループし、受信メッセージを順に `print`。タイムアウトは `[WS] (timeout waiting message)` として通知。
3. 例外発生時は `[WS] error: ...` を出力してループを抜け、最後に接続をクローズ。

## 入出力
- 入力: kabuステーションPUSH WebSocket。
- 出力: 標準出力へのメッセージログのみ。

## 実行例
```powershell
py scripts/ws_probe.py -Config config/stream_settings.json -Seconds 20 -Verbose 1
```
