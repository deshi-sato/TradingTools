# scripts/summarize_wf_results.py 仕様書

## 概要
Summarize walk-forward results under data/analysis.

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- なし

## 出力
- なし

## 設定項目
- OUTDIR: os.path.join("data", "analysis")

## 処理フロー
- 起動: __main__ ブロックあり
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- find_tags() -> Dict[str, Tuple[str, Optional[str]]]: Return mapping tag -> (results_csv, daily_csv_or_None).
- max_drawdown(cum_series: pd.Series) -> float: Return max drawdown (min of cumulative/rolling_max - 1). <= 0.
- metrics_for_tag(tag: str, res_csv: str, daily_csv: Optional[str]) -> Optional[dict]: 説明なし
- sort_tag_key(tag: str) -> Tuple[int, int]: 説明なし
- load_pairs() -> Dict[str, Tuple[str, Optional[str]]]: 説明なし
- main() -> int: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
