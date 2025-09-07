# scripts/fetch_yf_daily.py 仕様書

## 概要
TOPIX100 の日足を yfinance から増分取得し、SQLite に UPSERT します。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: path
- DB接続: sqlite3 args.db
- DB接続: sqlite3 args.db
- DB接続: sqlite3 db_path
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- JST: ZoneInfo("Asia/Tokyo")

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- 入出力: ファイルの読み書きを実施
- データアクセス: sqlite3 に接続・操作
- ロギング: logging による実行ログ出力
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- parse_args() -> argparse.Namespace: 説明なし
- ensure_schema(conn: sqlite3.Connection) -> None: 説明なし
- _normalize_ticker(code: str) -> str: 説明なし
- load_tickers_from_file(path: str) -> List[str]: 説明なし
- load_default_topix100() -> List[str]: 説明なし
- jst_today() -> date: 説明なし
- compute_end_date_exclusive() -> Tuple[date, date]: Returns (end_inclusive, end_exclusive) where inclusive is yesterday in JST.
- parse_date(s: str) -> date: 説明なし
- get_db_max_date(conn: sqlite3.Connection, ticker: str) -> Optional[date]: 説明なし
- history_df(ticker: str, start_date: date, end_exclusive: date) -> pd.DataFrame: 説明なし
- to_rows(ticker: str, df: pd.DataFrame) -> List[Tuple[str, str, float, float, float, float, int]]: 説明なし
- upsert_rows(conn: sqlite3.Connection, rows: Sequence[Tuple[str, str, float, float, float, float, int]]) -> int: 説明なし
- worker(db_path: str, task: Task, dry_run: bool) -> Tuple[str, int, Optional[str]]: Fetch and upsert one ticker. Returns (ticker, upserted_count, error_message).
- main() -> None: 説明なし
- Task: 説明なし

## 代表的なエラー
- Exception

## ログ
- 使用箇所: logging.basicConfig, error, info, warning

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
