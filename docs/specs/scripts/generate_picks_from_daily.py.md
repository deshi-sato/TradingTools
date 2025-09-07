# scripts/generate_picks_from_daily.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 db_path
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- なし

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- read_daily(db_path: str) -> pd.DataFrame: 説明なし
- rsi(series: pd.Series, n = 3) -> pd.Series: 説明なし
- add_features(g: pd.DataFrame) -> pd.DataFrame: 説明なし
- score_buy_row(r: pd.Series) -> float: 説明なし
- score_sell_row(r: pd.Series) -> float: 説明なし
- trend_up_idx(idx_row: pd.DataFrame) -> bool: 説明なし
- trend_down_idx(idx_row: pd.DataFrame) -> bool: 説明なし
- generate_picks(df: pd.DataFrame, start: date, end: date, out_dir: Path, min_turnover: float, topn: int, index_ticker: str|None, disable_sell_in_uptrend: bool, buy_overbought: float, sell_oversold: float, upper_wick_ratio_thr: float, lower_wick_ratio_thr: float, w_trend = None, w_volume = None, w_momo = None, single_out_csv = None) -> None: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- なし

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
