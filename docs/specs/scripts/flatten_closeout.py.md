# scripts/flatten_closeout.py 仕様書

## 概要
- kabuステーション REST API を用いて未約定注文のキャンセルと保有ポジションのクローズ注文を一括発行するツール。
- APIレート制御を内蔵し、1リクエスト500ms間隔・1分あたり上限などを守りながら安全にドライラン／本番実行を切り替えられる。

## 主な機能
- `/orders` `/positions` `/cancelorder` `/sendorder` を順番に呼び出し、アクティブ注文のキャンセルとテンプレートに基づくクローズ注文生成を自動化。
- `RateLimiter` で最小インターバルと1分あたり送信数を制御し、API制限に抵触しないよう待機。
- `flatten.order_template` に記載された注文テンプレートを読み込み、ポジションごとの `Symbol` `Exchange` `Side` `Qty` を埋めて送信。
- `--dry` 指定でHTTP送信をスキップし、ログだけで動作確認可能。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`host`/`port`/`token` と `flatten.order_template` を含むJSON。|
| `-Verbose` | ログレベル（1=INFO、0=WARNING）。|
| `--dry` | API呼び出しを抑止し、ログ出力のみ行う。|
| `--interval-ms` | RateLimiterの最小間隔（ms）。既定500。|
| `--per-minute` | 1分あたり許容リクエスト数。既定90。|

## 設定項目
| キー | 説明 |
|------|------|
| `host` / `port` | kabu API の接続先（通常 `localhost:18080`）。|
| `token` | X-API-KEY として使用するトークン。|
| `flatten.order_template` | `sendorder` に渡すベースJSON。銘柄・枚数などが上書きされる。|

## 処理フロー
1. 設定を読み込み、ログとレートリミッターを初期化。
2. `/orders` でアクティブ注文を取得し、`--dry` でなければ1件ずつ `cancelorder` を呼び出す。
3. `/positions` で保有ポジションを取得し、数量>0のみテンプレートをコピーして `_make_close_order_from_template` で整形。
4. ドライランでなければ `sendorder` を発行し、レスポンスコード・本文をログ出力。
5. すべての処理が完了したらINFOログで `done` を記録して終了。

## RESTアクセス
- HTTP通信は `urllib.request` を利用。JSON応答を解析しつつ、失敗時はテキストをそのまま保持してログに出力。
- 一時的なエラー（429/503/0）には指数バックオフで再試行。

## 実行例
```powershell
py scripts/flatten_closeout.py -Config config/stream_settings.json --dry -Verbose 1
py scripts/flatten_closeout.py -Config config/stream_settings.json --interval-ms 800 --per-minute 60
```
