# scripts/push_ws.py 仕様書

## 概要
- kabuステーションのWebSocket `/kabusapi/websocket` へ接続し、一定時間受信メッセージを標準出力へダンプする疎通確認ツール。
- X-API-KEYのみを指定すれば利用でき、PUSH配信の生データを素早く確認したいときに使う。

## 主な機能
- WebSocket接続の開始・終了時刻を表示し、受信したJSONを軽量整形してログ出力。
- 価格フィールド（`CurrentPrice`/`Price`/`BidPrice`/`AskPrice`）を抽出して簡易サマリを表示し、解析しやすくする。
- 例外時には`[ERROR]`行で例外内容を通知し、Ctrl+C で安全に停止可能。

## 主な引数
| 引数 | 説明 |
|------|------|
| `--host` | 接続先ホスト。既定 `localhost`。|
| `--port` | 接続先ポート。既定 18080。|
| `--token` | 必須。WebSocket用のAPIトークン。|
| `--seconds` | 受信を継続する目安秒数（待機ループに影響）。既定30。|

## 処理フロー
1. 引数を解釈し、WebSocket URL と `X-API-KEY` ヘッダを組み立てる。
2. `websocket.WebSocketApp` を作成し、`on_message` で受信JSONを解析して種別・銘柄・価格をログ。
3. `run_forever()` で接続を維持し、Ctrl+C で終了またはタイムアウト等でクローズする。
4. 受信中に例外が発生した場合は `[ERROR]` ログを出力し、終了時に `[CLOSE]` 行でコードと理由を表示。

## 入出力
- 入力: kabuステーション PUSH WebSocket。
- 出力: 標準出力へのテキストログのみ（ファイル出力なし）。

## 実行例
```powershell
py scripts/push_ws.py --token $env:KABU_TOKEN --seconds 60
py scripts/push_ws.py --host localhost --port 18081 --token ABCDEF123456
```
