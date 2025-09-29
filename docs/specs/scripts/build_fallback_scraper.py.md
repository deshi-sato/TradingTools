# scripts/build_fallback_scraper.py 仕様書

## 概要
- 松井証券のデイトレランキング（寄り付き前後）をスクレイピングし、`fallback_daytrade_YYYYMMDDHHMM.csv` を生成するバックアップ用データパイプライン。
- `requests` による静的HTML取得と、Selenium + Chromeによる動的取得を切り替えられ、環境に応じたフェッチ方法を選択可能。
- 取得した銘柄コードにフィルタを適用し、必要に応じて銘柄詳細ページから正式名称を補完してCSVに書き出す。

## 主な機能
- `config/fallback_filter.json` 等から除外コード・キーワード・価格帯フィルタを読み込み、`pass_filter()` で銘柄を足切りする。
- HTMLからランキングテーブルを抽出するパス (`parse_listing_rows_from_html`) と、Selenium DOM解析 (`parse_listing_rows_selenium`) を両立。
- 詳細ページの `og:title` から正式名称を取得し、ThreadPoolExecutor で並列解決。
- 失敗時は `data/fallback_daytrade_*.csv` の最新ファイルを探してフォールバックする安全策を内蔵。
- BOM付きUTF-8でCSVを書き出し、Excel連携時の文字化けを防止。

## 主な引数
| 引数 | 説明 |
|------|------|
| `--outdir` | 出力先ディレクトリ。既定は `data/`。|
| `--filter` | 除外条件や価格帯を定義したJSONパス。既定 `config/fallback_filter.json`。|
| `--encoding` | CSVの文字コード。既定 `utf-8-sig`（Excel互換）。|
| `--workers` | 銘柄名補完時の並列ワーカー数。既定10。|
| `--no-browser` | `requests` のみでランキングHTMLを取得し、Selenium/Chromeを起動しない。|
| `--debug` | ログレベルをDEBUGに引き上げ、解析状況を詳細出力。|

## 処理フロー
1. コマンドライン引数を解釈し、フィルタ設定を読み込む。
2. `--no-browser` の有無で取得モードを切り替え、ランキングページをページ送りしながらスクレイピングする。
3. 取得した行をフィルタに通し、並列で銘柄名称を補完したうえで `[code,name,reason]` 形式のリストへ整形。
4. 結果が空なら例外を投げ、既存の `fallback_daytrade_*.csv` をフォールバックとして採用する。
5. 正常時はサイトのタイムスタンプ（取得できない場合は現在時刻）でファイル名を組み立て、CSVを書き出す。
6. 生成状況を標準出力 (`summary fetched=...`) に記録し、Seleniumドライバをクリーンアップ。

## 入出力
- 入力: 松井証券ランキングHTML、フィルタJSON、過去に生成した `fallback_daytrade_*.csv`。
- 出力: `data/fallback_daytrade_<timestamp>.csv`（列: `code,name,reason`）。
- ログ: 進行状況とフォールバック判定を標準出力および標準エラーで通知。

## 外部依存
- `requests` / `beautifulsoup4` によるHTML解析。
- `selenium` + ChromeDriver（`--no-browser` 時は不要）。
- マルチスレッドで名称補完を行うため `concurrent.futures` を使用。

## 実行例
```powershell
py scripts/build_fallback_scraper.py --no-browser --encoding cp932 --debug
py -m scripts.build_fallback_scraper `
  --outdir data `
  --filter config/fallback_filter.json `
  --workers 12 `
  --no-browser

```
