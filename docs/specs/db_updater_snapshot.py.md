# db_updater_snapshot.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 str(DB_PATH)

## 出力
- なし

## 設定項目
- SCRIPT_DIR: Path(__file__).resolve().parent
- EXCEL_PATH: SCRIPT_DIR / "stock_data.xlsm"
- DB_PATH: SCRIPT_DIR / "data" / "rss_snapshot.db"
- DDL: "\n-- 当日（シート内の「最新日付」のみ）を保持する1分足\nCREATE TABLE IF NOT EXISTS today_data(\n  ticker     TEXT NOT NULL,\n  sheet_name TEXT NOT NULL,\n  datetime   TEXT NOT NULL,   -- 'YYYY-MM-DD HH:MM:SS' (JST)\n  open REAL, high REAL, low REAL, close REAL, volume INTEGER,\n  PRIMARY KEY(ticker, datetime)\n);\nCREATE INDEX IF NOT EXISTS idx_today_t_d ON today_data(ticker, datetime);\n\n-- シート先頭の気配スナップショット（参考）\nCREATE TABLE IF NOT EXISTS quote_latest(\n  ticker TEXT NOT NULL PRIMARY KEY,\n  sheet_name TEXT NOT NULL,\n  last REAL, prev_close REAL, open REAL, high REAL, low REAL,\n  volume INTEGER, turnover INTEGER, diff REAL, diff_pct REAL,\n  updated_at TEXT NOT NULL\n);\n"

## 処理フロー
- 起動: __main__ ブロックあり
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- _to_float(x) -> None: 説明なし
- _to_int(x) -> None: 説明なし
- _timestamp_from_time_cell(tval, base_dt: datetime) -> str: 説明なし
- extract_ticker_from_a1(a1: str) -> str: 説明なし
- init_db() -> sqlite3.Connection: 説明なし
- is_marketspeed_running() -> bool: 説明なし
- get_last_datetime(conn: sqlite3.Connection, ticker: str) -> str | None: 説明なし
- save_today_data(conn: sqlite3.Connection, ticker: str, sheet_name: str, df: pd.DataFrame) -> None: 説明なし
- read_code_sheet(ws: Worksheet) -> pd.DataFrame: 説明なし
- latest_day_only(df: pd.DataFrame) -> pd.DataFrame: シート内に混在しても「最新日付のみ」を返す
- read_snapshot(ws: Worksheet) -> dict: 説明なし
- upsert_quote_latest(conn: sqlite3.Connection, ticker: str, sheet_name: str, snap: dict) -> None: 説明なし
- get_targets(wb) -> List[Tuple[str, str]]: indexシート A1〜A30 に 4桁コード、対応する code1..code30 シートを対象とする
- cleanup_to_latest_date(conn: sqlite3.Connection, ticker: str) -> None: 保険：ticker単位で『最新日付のみ』をDBに残す（混入防止）
- clear_today_data(conn: sqlite3.Connection) -> None: 起動時に today_data をクリア
- main_loop(conn: sqlite3.Connection) -> None: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
