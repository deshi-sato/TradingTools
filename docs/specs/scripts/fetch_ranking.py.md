# scripts/fetch_ranking.py 仕様書

## 概要
- kabuステーションのランキングAPIを呼び出し、複数カテゴリを集約して `perma_regulars_YYYYMMDDHHMM.csv` を生成する現行版フェッチャー。
- `exec.api_client` が利用可能な場合はそちらを優先し、未導入環境ではローカルHTTPフォールバックで直接RESTへアクセスする。

## 主な機能
- `RANK_PLAN` に基づき出来高・売買代金・値上がり/値下がりランキングを取得し、理由タグを付けてリスト化。
- 重複銘柄を `dedup_rankings()` で統合し、理由タグをソートしたセミコロン区切りでまとめる。
- トークンが失効した場合に `/token` へPOSTして再発行し、環境変数 `KABU_TOKEN` / `KABU_API_KEY` を更新。
- 取得失敗時は標準エラーへ警告を出しつつシンボルをスキップし、最終的に0件であれば非ゼロ終了コードを返す。

## 主な引数
| 引数 | 説明 |
|------|------|
| `--outdir` | 出力ディレクトリ。既定 `data`。|
| `--encoding` | 出力CSVのエンコーディング。既定 `utf-8-sig`。|

## 環境変数
| 変数 | 役割 |
|------|------|
| `KABU_BASE_URL` | APIベースURL（既定 `http://localhost:18080`）。|
| `KABU_HTTP_TIMEOUT_POST` | HTTPタイムアウト秒。|
| `KABU_API_PW` / `KABU_API_PASSWORD` | `/token` 取得に使うパスワード。|
| `KABU_TOKEN` / `KABU_API_KEY` | `/ranking` 呼び出し時のX-API-KEY。必要に応じて書き換え。|
| `RANK_EXCHANGE` | `/ranking` の `ExchangeDivision` を上書き（既定 `ALL`）。|

## 処理フロー
1. コマンドライン引数と環境変数を読み込み、出力先とHTTP設定を決定。
2. `fetch_plan()` で `RANK_PLAN` の各カテゴリについて `get_ranking()` を呼び出し、失敗時は警告を記録。
3. 収集した `(symbol, exchange, reason)` を `dedup_rankings()` でマージし、理由タグをセミコロン結合したリストに整形。
4. 取得件数が0の場合はエラー終了。正常時は `perma_regulars_<timestamp>.csv` をUTF-8(BOM)で書き出す。
5. 標準出力に生成ファイルパスと件数を表示。

## 入出力
- 入力: kabuステーション `/kabusapi/ranking`、必要に応じて `/kabusapi/token`。
- 出力: `symbol,exchange,reason` 列のCSV。

## 連携
- `scripts/build_watchlist.py` が `perma_regulars_*.csv` を選択してウォッチリストを組み立てる。
- `scripts/build_fallback_scraper.py` のフォールバック入力としても利用可能。

## 実行例
```powershell
py scripts/fetch_ranking.py --outdir data
KABU_API_PW=xxxx py scripts/fetch_ranking.py --encoding cp932
```
