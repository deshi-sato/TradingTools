# export_minutes_to_csv.py 仕様書

## 概要
Export minute bars from sqlite3 to per-symbol CSV files.

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 path
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- ファイル出力: out_path mode='w'

## 設定項目
- なし

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- 入出力: ファイルの読み書きを実施
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- parse_args() -> argparse.Namespace: 説明なし
- friendly_exit(msg: str, code: int = 1) -> None: 説明なし
- ensure_iso8601(ts: object) -> str: Best-effort conversion to ISO8601 string with seconds precision.
- open_db(path: str) -> sqlite3.Connection: 説明なし
- build_query(symbols: Optional[List[str]], start: str, end: str) -> Tuple[str, List[object]]: 説明なし
- export_rows(conn: sqlite3.Connection, out_dir: str, sql: str, params: List[object]) -> int: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- Exception
- KeyboardInterrupt
- sqlite3.Error
- sqlite3.OperationalError

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
