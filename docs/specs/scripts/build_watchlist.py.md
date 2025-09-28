# scripts/build_watchlist.py 仕様書

## 概要
- `data/perma_regulars_*.csv` と `data/fallback_daytrade_*.csv` を走査し、最新のランキングを元に `watchlist_today.csv` を生成するユーティリティ。
- CSVの文字コード差異を吸収しつつ `symbol` / `exchange` の2列に正規化して書き出す。
- スケジュールジョブやPowerShellランナーから起動し、当日のウォッチリストを自動更新する用途を想定。

## 主な機能
- UTF-8(BOM) / UTF-8 / CP932 を順に試す `read_csv_any()` で入力CSVを安定して読み込む。
- `perma_regulars` と `fallback_daytrade` のタイムスタンプ付きファイル名から最新候補を選択し、必要なら `--force-fallback` で強制切り替え。
- 行毎にカラム名の揺れ (`symbol`/`code`/`Code` 等) を解決し、欠損行を除外して `(symbol, exchange)` タプルへ変換。
- `--limit` 指定で上位N件のみを採用し、UTF-8(BOM) でウォッチリストを書き出す。
- 主要イベントを `logging` でINFO/DEBUG出力し、異常時はエラーコードで終了。

## 主な引数
| 引数 | 説明 |
|------|------|
| `--data-dir` | 元データ（perma/fallback）が置かれたディレクトリ。既定 `.\data`。|
| `--output` | 出力するウォッチリストCSVパス。既定 `.\data\watchlist_today.csv`。|
| `--limit` | 出力件数の上限。指定がない場合は全件。|
| `--force-fallback` | 常に `fallback_daytrade_*.csv` を選択し、perma_regulars を無視。|
| `--debug` | ログレベルをDEBUGに変更。|

## 処理フロー
1. 引数を解析し、ロガーを初期化。
2. `--force-fallback` の有無に応じ、`perma_regulars_*.csv` / `fallback_daytrade_*.csv` を最新タイムスタンプで選択。
3. 選択したCSVを `read_csv_any()` で読み込み、`normalize_to_symbol()` で `(symbol, exchange)` リストに整形。
4. `--limit` が設定されていれば先頭からN件にトリミング。
5. `write_watchlist()` で `symbol,exchange` の2列CSVをUTF-8(BOM)で書き出し、処理件数をINFOログに残す。
6. 成功時は0、異常時は1を返して終了。

## 入出力
- 入力: `data/perma_regulars_*.csv` または `data/fallback_daytrade_*.csv`。
- 出力: `watchlist_today.csv`（UTF-8(BOM), 列: `symbol,exchange`）。
- ログ: `INFO`/`DEBUG`で処理状況を標準出力に記録。

## 連携
- `scripts/build_fallback_scraper.py` や `scripts/fetch_ranking.py` が生成するランキングCSVを消費。
- 出力した `watchlist_today.csv` は `scripts/register_watchlist.py` で PUSH登録に利用される。

## 実行例
```powershell
py scripts/build_watchlist.py --data-dir .\data --output .\data\watchlist_today.csv
py scripts/build_watchlist.py --force-fallback --limit 40 --debug
```
