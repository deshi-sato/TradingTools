# scripts/kabus_login_wait.py 仕様書

## 概要
- kabuステーションがログイン完了し `/token` API が利用可能になるまでリトライし、取得したトークンを設定ファイルへ書き戻す補助スクリプト。
- 最大10回、3秒置きにポーリングしてトークン発行を待ち受ける。

## 主な機能
- JSON設定から `port` と `api_password` を読み込み、PowerShell等からの呼び出しで再利用。
- `/kabusapi/token` に対してPOSTを発行し、取得した `Token` を `config` の `token` フィールドへ保存。
- 成功時/失敗時のレスポンス内容を標準出力へ整形して表示し、最終的に例外で失敗通知。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`port` と `api_password` を含む設定ファイル。|

## 処理フロー
1. 設定を読み込み、`api_password` が存在しなければ即座に例外を送出。
2. `/kabusapi/token` へ最大10回POSTし、`Token` を含む応答を待つ。状態コードやJSON内容を都度表示。
3. トークン取得に成功したら設定ファイルへ書き戻し、メッセージを出力して終了。
4. 取得できないままリトライ上限に達した場合は `RuntimeError` を投げる。

## 入出力
- 入力: 設定JSON、APIパスワード。
- 出力: 更新済みの設定JSON（`token` フィールドが追加）。
- ログ: 標準出力に進捗とエラーを表示。

## 実行例
```powershell
py -m scripts.kabus_login_wait -Config config\stream_settings.json
```
