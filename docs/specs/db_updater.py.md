# db_updater.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 str(db_path)

## 出力
- なし

## 設定項目
- SCRIPT_DIR: Path(__file__).resolve().parent
- EXCEL_PATH: SCRIPT_DIR / "デイトレ株価データ.xlsm"
- DB_PATH: SCRIPT_DIR / "data" / "rss_data.db"
- DDL: "\n-- 1分足\nCREATE TABLE IF NOT EXISTS minute_data (\n    ticker      TEXT    NOT NULL,   -- 銘柄コード(16進4桁・大文字)\n    sheet_name  TEXT    NOT NULL,   -- 銘柄名称（シート名）\n    datetime    TEXT    NOT NULL,   -- 'YYYY-MM-DD HH:MM:SS' (JST)\n    open        REAL,\n    high        REAL,\n    low         REAL,\n    close       REAL,\n    volume      INTEGER,\n    PRIMARY KEY (ticker, datetime)\n);\nCREATE INDEX IF NOT EXISTS idx_minute_data_ticker_datetime\n  ON minute_data (ticker, datetime);\n\n-- 最新スナップショット（銘柄ごとに1行）\nCREATE TABLE IF NOT EXISTS quote_latest (\n  ticker       TEXT    NOT NULL,      -- 16進4桁（大文字）\n  sheet_name   TEXT    NOT NULL,      -- シート名（銘柄名）\n  last         REAL,                  -- 現在値\n  prev_close   REAL,                  -- 前日終値\n  open         REAL,\n  high         REAL,\n  low          REAL,\n  volume       INTEGER,               -- 出来高（累計）\n  turnover     INTEGER,               -- 売買代金（累計）\n  diff         REAL,                  -- 前日比\n  diff_pct     REAL,                  -- 前日比率[%]\n  updated_at   TEXT    NOT NULL,      -- 'YYYY-MM-DD HH:MM:SS' (JST)\n  PRIMARY KEY (ticker)\n);\nCREATE INDEX IF NOT EXISTS idx_quote_latest_updated_at\n  ON quote_latest (updated_at);\n"

## 処理フロー
- 起動: __main__ ブロックあり
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- _to_float(x) -> None: 数値/数値文字列→float（それ以外は None）
- _to_int(x) -> None: 数値/数値文字列→int（それ以外は None）
- _timestamp_from_time_cell(tval, base_dt: datetime) -> str: R2（時刻セル）から 'YYYY-MM-DD HH:MM:SS' を作る。無ければ base_dt を使う。
- extract_hex_ticker_from_a1(a1: str) -> str: =RssChart(..., "285A.T", ...) から 16進4桁コードを抽出。返り値は大文字。
- init_db(db_path: Path = DB_PATH) -> sqlite3.Connection: 説明なし
- get_last_datetime(conn: sqlite3.Connection, ticker: str) -> str | None: 説明なし
- save_minute_data(conn: sqlite3.Connection, ticker: str, sheet_name: str, df: pd.DataFrame) -> None: DataFrame → minute_data へ差し込み（Pythonの型に正規化）
- read_excel_fixed(path: Path) -> Dict[Tuple[str, str], pd.DataFrame]: 固定仕様：
- read_snapshot_with_map(ws: Worksheet) -> dict: SNAPSHOT_MAP に基づきセルを読み、必要な補完を行って dict を返す。
- upsert_quote_latest(conn: sqlite3.Connection, ticker: str, sheet_name: str, snap: dict) -> None: quote_latest を UPSERT（ticker ごとに最新1行保持）
- is_marketspeed_running_cmd() -> None: 説明なし
- main_loop() -> None: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
