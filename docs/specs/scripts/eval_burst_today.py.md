# scripts/eval_burst_today.py 仕様書

## 概要
- `features_stream` テーブルに蓄積されたバースト指標を抽出し、ティッカー別の発生回数と平均スコアを集計するワンショットレポート。
- 特定日時以降や特定ティッカーに絞り込むことで、日次監視や閾値チューニング時の確認を容易にする。

## 主な機能
- JSON設定からSQLiteパスを読み込み、`features_stream` を直接SQLで参照。
- `-Since` でISO日時以降、`-Ticker` で対象銘柄をフィルタできる柔軟なWHERE句生成。
- `burst_buy`/`burst_sell` フラグと `burst_score` を集計し、CSV形式で標準出力へ結果を吐き出す。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`db_path` を含むJSON設定ファイル。|
| `-Since` | 任意。`YYYY-MM-DD HH:MM` 形式の閾値。指定日時以降のみ集計。|
| `-Ticker` | 任意。単一ティッカーに限定。|

## 処理フロー
1. 引数を解析し、`config` からSQLiteパスを取得。
2. `features_stream` へのSELECT文を組み立て、`burst_*` 列と補助指標を取得。
3. Python側で閾値別に集約し、BUY/Sellの発生数とスコア合計を保持。
4. ティッカーごとに平均値を算出し、`ticker,buy_cnt,sell_cnt,avg_burst_score` をヘッダー付きで出力。

## 入出力
- 入力: `config/stream_settings.json` 等に含まれる `db_path`（SQLite DB）。
- 出力: 標準出力へのCSV行（ファイル出力は行わない）。

## 実行例
```powershell
py scripts/eval_burst_today.py -Config config/stream_settings.json -Since "2025-09-26 09:00" > out/burst_eval.csv
py scripts/eval_burst_today.py -Config config/stream_settings.json -Ticker 7203
```
