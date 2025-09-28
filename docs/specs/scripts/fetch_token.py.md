# scripts/fetch_token.py 仕様書

## 概要
- kabuステーションの `/token` API を叩いて新しいAPIトークンを取得し、設定JSON (`kabu.api_token`) を更新するワンショットツール。
- CLIでAPIパスワードを受け取り、トークン再発行の手作業を簡略化する。

## 主な機能
- UTF-8(BOM)対応の設定読込 (`load_json_utf8`) とHTTP POSTを組み合わせ、トークン取得の失敗理由を丁寧に報告。
- 取得したトークンを即座に設定JSONへ書き戻し、他ツールが再読み込みするだけで反映されるようにする。
- APIレスポンスが不正な場合や通信エラー時にはエラーメッセージを標準出力へ表示し終了コード1を返す。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`kabu.base_url` を含む設定ファイルパス。|
| `--pass` | 必須。APIパスワード。PowerShell等から安全に渡す。|

## 処理フロー
1. 設定ファイルを読み込み、`kabu.base_url` を取得（末尾スラッシュは自動調整）。
2. `/token` エンドポイントへ `{"APIPassword": ...}` をPOST。
3. HTTPエラーやJSON不備を検証し、`Token` フィールドを取り出す。存在しない場合はエラー終了。
4. `cfg["kabu"]["api_token"]` を新トークンで更新し、UTF-8で書き戻す。
5. 成功メッセージを表示し、終了コード0で終了。

## 入出力
- 入力: 設定JSON (`config/stream_settings.json` 等)、APIパスワード。
- 出力: 更新された設定ファイル（`api_token` が最新化）。
- 標準出力: 実行結果メッセージ。

## 実行例
```powershell
py scripts/fetch_token.py -Config config/stream_settings.json --pass (Get-Secret KabuApiPass)
```
