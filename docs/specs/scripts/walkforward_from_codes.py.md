# scripts/walkforward_from_codes.py 仕様書

## 概要
Walk-forward evaluator for daily code scores.

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- 環境変数: WF_TEST, WF_TRAIN

## 出力
- なし

## 設定項目
- CODES_CSV: 'data/score_daily.codes.csv'
- OUTDIR: 'data/analysis'
- TRAIN_DAYS: int(os.getenv("WF_TRAIN", "180"))
- TEST_DAYS: int(os.getenv("WF_TEST", "30"))
- N_LIST: [5, 10, 20]

## 処理フロー
- 起動: __main__ ブロックあり
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- daily_mean_ret(sub: pd.DataFrame, top: bool, n: int) -> float: Mean next_return for Top/Bottom N by score on a single day.
- window_eval(df: pd.DataFrame, train_dates: pd.Series) -> dict: Pick best of candidate strategies on train period (by mean return).
- apply_strategy(df: pd.DataFrame, test_dates: pd.Series, side: str, n: int) -> pd.DataFrame: Apply chosen strategy on test period and return daily returns.
- main() -> int: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
