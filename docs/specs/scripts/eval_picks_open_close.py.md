# scripts/eval_picks_open_close.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 args.db_path
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- DB_TABLE: 'daily_bars'
- ENCODINGS: ["utf-8-sig", "utf-8", "cp932"]

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- parse_args() -> None: 説明なし
- bps_to_frac(bps: float) -> float: 説明なし
- parse_date_from_filename(name: str) -> Optional[date]: 説明なし
- read_csv_flex(fp: Path) -> Optional[pd.DataFrame]: 説明なし
- find_col(cols: List[str], candidates: List[str]) -> Optional[str]: 説明なし
- get_price(conn, ticker: str, d: date) -> Optional[Tuple[float, float]]: 説明なし
- gather_pick_files(root: Path, recursive: bool) -> List[Path]: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
