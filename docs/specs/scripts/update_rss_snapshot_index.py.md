# scripts/update_rss_snapshot_index.py 仕様書

## 概要
watchlist_YYYY-MM-DD.csv の最新ファイルから、

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- DEFAULT_DATA_DIR: Path("./data")
- DEFAULT_EXCEL: DEFAULT_DATA_DIR / "stock_data.xlsm"
- DEFAULT_SHEET: 'index'
- DEFAULT_MAX_BUY: 15
- DEFAULT_MAX_SELL: 15
- WATCH_RE: re.compile(r"watchlist_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- ロギング: logging による実行ログ出力
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- setup_logger(log_path: str | None, level: str = 'INFO') -> logging.Logger: 説明なし
- pick_latest_watchlist(data_dir: Path) -> Path: ファイル名の日付を優先、なければ更新時刻で最新を選ぶ。
- read_codes_by_side(path: Path, max_buy: int, max_sell: int, logger: logging.Logger) -> Tuple[List[str], List[str]]: 説明なし
- write_to_xlsm(excel_path: Path, sheet: str, buys: List[str], sells: List[str], max_buy: int, max_sell: int, logger: logging.Logger) -> None: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- Exception

## ログ
- 使用箇所: logging.FileHandler, Formatter, StreamHandler, getLogger

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
