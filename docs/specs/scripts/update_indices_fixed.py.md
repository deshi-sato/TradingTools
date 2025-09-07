# scripts/update_indices_fixed.py 仕様書

## 概要
固定セルの指標データを Excel(.xlsm) から読み、SQLite に書き込む。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: args.config
- ファイル入力: path
- DB接続: sqlite3 db_path
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- DEFAULT_MAPPING: {
    "sheet": 1,
    "snapshots": [
        {"code": "DJIA",   "name": "NYダウ",           "date": "B3",  "time": "C3",  "last": "D3",  "pct": "E3"},
        {"code": "NASDAQ", "name": "NASDAQ総合指数",   "date": "B4",  "time": "C4",  "last": "D4",  "pct": "E4"},
        {"code": "SP500",  "name": "S&P500指数",       "date": "B5",  "time": "C5",  "last": "D5",  "pct": "E5"},
        {"code": "VIX",    "name": "VIX指数",          "date": "B6",  "time": "C6",  "last": "D6",  "pct": "E6"},
    ],
    "ohlcv": [
        {"code":"USDJPY","name":"ドル/円(Bid)","rows":100,
         "date":"H3","time":"I3","open":"J3","high":"K3","low":"L3","close":"M3","volume":"N3"},
        {"code":"N225","name":"日経225","rows":100,
         "date":"Q3","time":"R3","open":"S3","high":"T3","low":"U3","close":"V3","volume":"W3"},
        {"code":"N225_FUT","name":"225先物大阪(期近)","rows":100,
         "date":"Z3","time":"AA3","open":"AB3","high":"AC3","low":"AD3","close":"AE3","volume":"AF3"},
        {"code":"TOPIX","name":"TOPIX","rows":100,
         "date":"AI3","time":"AJ3","open":"AK3","high":"AL3","low":"AM3","close":"AN3","volume":"AO3"},
        {"code":"NikkeiVI","name":"日経平均VI指数","rows":100,
         "date":"AR3","time":"AS3","open":"AT3","high":"AU3","low":"AV3","close":"AW3","volume":"AX3"}
    ]
}
- DDL: "\nCREATE TABLE IF NOT EXISTS market_snapshots(\n  code TEXT NOT NULL,\n  datetime TEXT NOT NULL,     -- 'YYYY-MM-DD HH:MM:SS' JST\n  last REAL,\n  pct_change REAL,\n  source TEXT DEFAULT 'xlsm',\n  PRIMARY KEY(code, datetime)\n);\nCREATE TABLE IF NOT EXISTS market_ohlcv(\n  code TEXT NOT NULL,\n  datetime TEXT NOT NULL,     -- 'YYYY-MM-DD HH:MM:SS' JST\n  open REAL, high REAL, low REAL, close REAL, volume REAL,\n  source TEXT DEFAULT 'xlsm',\n  PRIMARY KEY(code, datetime)\n);\nCREATE INDEX IF NOT EXISTS idx_ms_code_dt ON market_snapshots(code, datetime);\nCREATE INDEX IF NOT EXISTS idx_mo_code_dt ON market_ohlcv(code, datetime);\n"
- EXCEL_EPOCH: datetime(1899, 12, 30)
- _HYPHEN_RE: re.compile(r"^-{3,}$")

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- 入出力: ファイルの読み書きを実施
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- coords_from_mapping(ws, it) -> None: it = {'date':'H3', 'time':'I3', 'open':'J3', ...} を行頭/各列indexに分解
- iter_rows_until_hyphen(ws, start_row, cols, max_rows = 2000) -> None: 行ベースで '-----' が来るまで (date,time,o,h,l,c,v) のタプルをyield
- is_hyphen_sentinel(x) -> bool: '-----' などの終端行か？
- read_col_until_sentinel(ws: Worksheet, start_addr_or_name: str, max_rows: int = 2000) -> None: start セルから下方向に、'-----' が出るまで読み続けてリストで返す。
- parse_excel_date(val) -> Optional[datetime]: 説明なし
- parse_excel_time(val) -> Optional[datetime]: 説明なし
- parse_percent(val) -> Optional[float]: 説明なし
- coordinate(addr: str) -> Tuple[int,int]: 説明なし
- resolve_sheet(wb, spec) -> None: 説明なし
- get_named_or_cell(ws: Worksheet, name_or_addr: str) -> None: 説明なし
- read_down(ws: Worksheet, start_addr_or_name: str, n: int) -> None: 説明なし
- read_file_safely(path: Path, stable_wait = 0.2, retries = 5, backoff = 0.3) -> bytes: 説明なし
- open_workbook_no_lock(path: Path) -> None: 説明なし
- is_hyphen_sentinel(x) -> bool: '-----' などの終端行か？ 
- dt_to_str(dt: datetime) -> str: 説明なし
- get_last_dt(con: sqlite3.Connection, code: str) -> Optional[datetime]: 説明なし
- import_snapshots(ws: Worksheet, con: sqlite3.Connection, items, mode: str) -> None: 説明なし
- import_ohlcv(ws: Worksheet, con: sqlite3.Connection, items, mode: str) -> None: 説明なし
- run_once(excel_path: Path, db_path: Path, mapping: Dict[str,Any], sheet_override = None, mode = 'replace') -> None: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- Exception
- KeyboardInterrupt

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
