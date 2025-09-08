# bench_nautilus_csv.py 仕様書

## 概要
CSV replay benchmark in a nautilus-like style.

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: path

## 出力
- なし

## 設定項目
- ENGINE_NAME: 'nautilus_like_csv_replay'

## 処理フロー
- 起動: __main__ ブロックあり
- 入出力: ファイルの読み書きを実施
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- friendly_exit(msg: str, code: int = 1) -> None: 説明なし
- load_close_series(path: str) -> Tuple[np.ndarray, np.ndarray]: Load timestamp and close arrays from a CSV file.
- simple_ma_cross_trades(close: np.ndarray, fast: int = 5, slow: int = 25) -> int: Count number of bullish crossovers (fast SMA crossing above slow SMA).
- main() -> None: 説明なし

## 代表的なエラー
- FileNotFoundError
- Tuple

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
