# signal_watcher.py 仕様書

## 概要
cls

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 args.snapshot_db
- DB接続: sqlite3 args.index_db
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- ファイル出力: log_path mode='a'

## 設定項目
- JST: tz.gettz("Asia/Tokyo")
- VOL_SPIKE_MULT: 1.5
- VWAP_MIN_BARS_BELOW: 3
- R1_MULT: 1.3
- R2_MULT: 2.0
- MIN_R_YEN: 5.0
- ORB_START: time(9, 0, 0)
- ORB_END: time(9, 15, 0)

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- 入出力: ファイルの読み書きを実施
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- pick_fallback_signal(base: pd.DataFrame, ticker: str, side: str, window_end: pd.Timestamp) -> Optional[Signal]: 締切時点のその銘柄の最新バー（<window_end）で、side向けのOCOを組んだフォールバックSignalを作る。
- consec_true(series: pd.Series) -> pd.Series: Trueが何本連続しているか（当該バーでの長さ）
- load_today_data(conn_snap: sqlite3.Connection, tickers: Optional[List[str]] = None, start_hhmm: str = '09:00') -> pd.DataFrame: 説明なし
- parse_args() -> argparse.Namespace: 説明なし
- ensure_dirs(path: str) -> None: 説明なし
- log(msg: str, log_path: str) -> None: 説明なし
- list_tables(conn: sqlite3.Connection) -> List[str]: 説明なし
- table_has_columns(conn: sqlite3.Connection, table: str, cols: List[str]) -> bool: 説明なし
- find_score_table(conn_index: sqlite3.Connection) -> Optional[str]: rss_index.db 内で 'ticker' と 'score_buy' or 'score_sell' を持つテーブルを探索。
- maybe_load_scores(conn_index: Optional[sqlite3.Connection]) -> pd.DataFrame: 説明なし
- find_watchlist(conn_snap: sqlite3.Connection) -> Optional[str]: snapshot側に watchlist的テーブルがあるなら使う（side列でBUY/SELLを事前指定できる）。
- compute_indicators(df: pd.DataFrame) -> pd.DataFrame: df: 単一tickerの当日1分足（ソート済）
- calc_orb_levels(df: pd.DataFrame) -> Tuple[float, float]: 説明なし
- last_cross_up(a: pd.Series, b: pd.Series) -> bool: 直近2本で a が b を上抜いたか？
- last_cross_down(a: pd.Series, b: pd.Series) -> bool: 直近2本で a が b を下抜いたか？
- bars_below(a: pd.Series, b: pd.Series, n: int) -> int: 直近連続して a<b だった本数を返す
- bars_above(a: pd.Series, b: pd.Series, n: int) -> int: 説明なし
- swing_low_recent(df: pd.DataFrame, lookback: int = 3) -> float: 説明なし
- swing_high_recent(df: pd.DataFrame, lookback: int = 3) -> float: 説明なし
- propose_oco(side: str, entry_price: float, ref_df: pd.DataFrame) -> Tuple[float, float, float, float, str]: 損切り/目標の自動提案
- detect_signals_one(df: pd.DataFrame, pre_buy: Optional[float], pre_sell: Optional[float]) -> List[Signal]: 単一ticker（当日分）でシグナル検出（バーごとに評価）。
- main() -> None: 説明なし
- Signal: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
