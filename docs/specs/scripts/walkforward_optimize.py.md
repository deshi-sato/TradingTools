# scripts/walkforward_optimize.py 仕様書

## 概要
Monthly walk-forward optimization for generate_picks_from_daily parameters.

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- SCRIPT_DIR: Path(__file__).parent

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- parse_args() -> argparse.Namespace: 説明なし
- month_start(d: date) -> date: 説明なし
- month_end(d: date) -> date: 説明なし
- add_months(d: date, months: int) -> date: 説明なし
- compute_features_full(df: pd.DataFrame) -> pd.DataFrame: 説明なし
- main() -> None: 説明なし
- FoldResult: 説明なし
-   - buy_rate(self) -> float: 説明なし
-   - sell_rate(self) -> float: 説明なし
-   - avg_rate(self) -> float: 説明なし

## 代表的なエラー
- なし

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
