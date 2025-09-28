# scripts/register_watchlist.py 仕様書

## 概要
- `watchlist_top50.csv` などのコード一覧を読み込み、kabuステーションの `/register` API にまとめて登録するユーティリティ。
- CSVの文字コード・区切り記号を自動判別し、`Symbol` と `Exchange` のセットに整形してPUTする。

## 主な機能
- `load_codes()` でUTF-8(BOM)・CP932 等を順に試しながらコード列を検出し、空行や重複を除外。
- `build_symbols()` でAPI形式 `{"Symbol": code, "Exchange": exch}` のリストを生成。
- `put_register()` が `/kabusapi/register` へPUTし、レスポンスのJSONをログ表示。`-DryRun` で送信を抑止。
- 主要操作を標準出力へログし、失敗時は非ゼロ終了コードで停止。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`port` と `token` を含む設定JSON。|
| `-Input` | 必須。登録対象CSV（`code` 列などを含む）。|
| `-Max` | 登録する最大件数。既定50。|
| `-Exchange` | 登録時にセットする取引所コード。既定1。|
| `-Verbose` | ログレベル（1=詳細、0=簡略）。|
| `-DryRun` | API送信を行わず、読み込んだ銘柄だけ表示。

## 処理フロー
1. 設定ファイルを読み込み、ポートとトークンを取得。欠損時は例外で終了。
2. 入力CSVをデコードし、`code`/`ticker` 等の列から銘柄コードを抽出。
3. `-Max` 件にトリムし、`build_symbols()` でAPIに適した形式へ整形。
4. `-DryRun` でなければ `/register` へPUTし、ステータスコードと本文キーをログ出力。
5. 応答が200以外の場合はstderrへ詳細を出力し、終了コード2/3で停止。

## 入出力
- 入力: CSV（列例: `code`, `symbol`, `ティッカー` など）。
- 出力: APIレスポンスのログのみ。ファイルは更新しない。

## 実行例
```powershell
py scripts/register_watchlist.py -Config config/stream_settings.json -Input data/watchlist_top50.csv
py scripts/register_watchlist.py -Config config/stream_settings.json -Input data/watchlist_today.csv -Max 30 -DryRun
```
