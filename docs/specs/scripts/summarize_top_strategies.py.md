# scripts/summarize_top_strategies.py 仕様書

## 概要
Pick top WF configs from wf_summary_table.csv with filters and sorting.

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- 環境変数: name

## 出力
- なし

## 設定項目
- OUTDIR: os.path.join("data", "analysis")
- TABLE: os.path.join(OUTDIR, "wf_summary_table.csv")
- TOP_CSV: os.path.join(OUTDIR, "wf_summary_top.csv")
- TOP_BAR: os.path.join(OUTDIR, "wf_summary_top_bar.png")

## 処理フロー
- 起動: __main__ ブロックあり
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- read_env_float(name: str, default: float) -> float: 説明なし
- read_env_int(name: str, default: int) -> int: 説明なし
- main() -> int: 説明なし

## 代表的なエラー
- Exception
- ValueError

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
