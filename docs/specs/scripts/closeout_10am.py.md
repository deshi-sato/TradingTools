# scripts/closeout_10am.py 仕様書

## 概要
- 当日の `logs/orders-YYYYMMDD.jsonl` を集計し、10時クローズ時点の注文・ポジション状況をサマリ化するスクリプト。
- `MODE=LIVE` 環境では kabuステーション実行系 (`exec.kabu_exec`) を呼び出して全注文キャンセルと保有ポジションのクローズを試みる。
- 処理結果をJSONで保存し、`logs/` と `data/` 配下の当日ファイルをZIPにまとめてアーカイブする。

## 主な機能
- `collect_orders()` で当日分の注文ログJSONLを読み込み、シンボル集合やIFDOCO件数を抽出。
- `summarize()` でBUY/SELL比率やリジェクト件数を集計し、実行モード (`MODE`) を付与。
- LIVEモード時に `exec.kabu_exec` の `cancel_all_orders` / `close_all_positions` を順に呼び出し、成功可否をログ。
- `zip_logs()` で `logs/*YYYYMMDD*` および `data/*YYYYMMDD*` を `archive/YYYYMMDD.zip` に保存。

## 環境変数
| 変数 | 説明 |
|------|------|
| `MODE` | `LIVE` の場合のみ実際の注文取消し・ポジション解消を試行。既定は `PAPER`。|

## 処理フロー
1. 現地日付（JST）のタグを生成し、`logs/orders-<tag>.jsonl` をすべて読み込む。
2. 注文ログから銘柄一覧・IFDOCO数・リジェクト数・BUY/SELL件数などを集計。
3. 環境変数 `MODE` を読み取り、LIVEなら `try_live_close()` でAPI呼び出しを試行。
4. サマリ結果を `logs/close_summary-<tag>.json` に書き出す。
5. `archive/<tag>.zip` を生成し、`logs/` / `data/` の該当ファイルを詰める。
6. 進捗を標準出力へログし、異常時は標準エラーへ出力して非ゼロ終了コードで停止。

## 入出力
- 入力: `logs/orders-YYYYMMDD.jsonl`（新規約定ループが生成するイベントログ）。
- 出力: `logs/close_summary-YYYYMMDD.json` と `archive/YYYYMMDD.zip`。
- 標準出力/標準エラーでアクション結果を報告。

## 連携
- `py orchestrate/run_intraday.py` が吐き出す注文ログを集計対象に想定。
- LIVEモードでは `exec/kabu_exec.py` 内のラッパー関数に依存。

## 実行例
```powershell
MODE=PAPER py scripts/closeout_10am.py
MODE=LIVE  py scripts/closeout_10am.py
```
