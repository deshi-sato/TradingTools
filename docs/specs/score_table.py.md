# score_table.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 db_path

## 出力
- なし

## 設定項目
- なし

## 処理フロー
- 起動: __main__ ブロックなし
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- save_chart_5min(ticker, df, global_data_dict) -> None: 説明なし
- evaluate_stock_long(day_frames) -> None: 説明なし
- evaluate_stock_short(day_frames) -> None: 説明なし
- create_score_table_long(data_dict) -> None: 説明なし
- create_score_table_short(data_dict) -> None: 説明なし
- _latest_trade_date(conn: sqlite3.Connection, ticker: str) -> Optional[str]: 説明なし
- _latest_completed_trade_date(conn: sqlite3.Connection, ticker: str, min_bars: int = 332) -> Optional[str]: 説明なし
- _count_minutes_of_day(conn: sqlite3.Connection, ticker: str, trade_date: str) -> int: 説明なし
- _daily_ohlcv(conn: sqlite3.Connection, ticker: str, trade_date: str) -> Optional[Dict[str, float]]: 説明なし
- _prev_daily_refs(conn: sqlite3.Connection, ticker: str, base_date: str, n_days: int = 5, min_bars: int = 332) -> List[Dict[str, float]]: 説明なし
- compute_trend_score_for_snapshots(db_path: str) -> Dict[str, Optional[int]]: Return {ticker: score or None} for all tickers in quote_latest.
- _score_buy(today, prevs) -> int: 翌日上昇シグナル向けの買いスコアを算出（当日＋直近過去データの簡易日足ベース）。
- _score_sell(today, prevs) -> int: 翌日下降シグナル向けの売りスコアを算出（当日＋直近過去データの簡易日足ベース）。
- _slope_last(series: pd.Series, span: int) -> float: 直近 span+1 区間での単純傾き（終値ベース）。データ不足時は0。
- add_trend_score(df: pd.DataFrame, ma_fast: int = 5, ma_slow: int = 25, slope_fast_span: int = 5, slope_slow_span: int = 10, breakout_lookback: int = 3) -> pd.DataFrame: df: 必須列 -> ['close','high','low']、インデックス or 列に日付があること

## 代表的なエラー
- Exception
- Tuple

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
