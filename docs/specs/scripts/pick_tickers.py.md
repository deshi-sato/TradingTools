# scripts/pick_tickers.py 仕様書

## 概要
戦略タグ (T{window}_V{volma}) を実銘柄へ落とす。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 args.db
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- TAG_RE: re.compile(r"^T(?P<T>\d+)_V(?P<V>\d+)$")

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- データアクセス: sqlite3 に接続・操作
- ロギング: logging による実行ログ出力
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- setup_logger(log_path: str | None, level: str = 'INFO') -> logging.Logger: 説明なし
- parse_tag(tag: str) -> tuple[int, int]: 説明なし
- latest_asof(conn: sqlite3.Connection) -> str: 説明なし
- trend_slope(prices: pd.Series) -> float: 説明なし
- compute_atr(df: pd.DataFrame, window: int) -> float: df: 必須列 ['high','low','close']、日付昇順で直近まで含む想定。
- compute_sigma(close: pd.Series, T: int) -> float: 直近T本の logリターン標準偏差（≒日次σ）。
- safe_floor(x: float) -> int: 説明なし
- round_lot(size: int, lot: int = 100) -> int: 売買単位で丸め（例: 100株単位）。0未満は0。
- build_base_table(conn: sqlite3.Connection, tag: str, asof: str, min_vol_ma: float, min_days: int | None, size_mode: str, capital: float, risk_pct: float, atr_window: int, atr_mult: float, vol_target_pct: float, lot: int, min_notional: float, max_notional: float) -> pd.DataFrame: 全銘柄について指標・サイズを計算し、「eligible/reason」を付けた明細テーブル(DataFrame)を返す。
- choose_buy_sell(base: pd.DataFrame, tag: str, asof: str, top_long: int, top_short: int) -> pd.DataFrame: eligible==True のみから BUY/SELL を抽出し、最終の出力フォーマットで返す。
- main() -> None: 説明なし

## 代表的なエラー
- Exception

## ログ
- 使用箇所: logging.FileHandler, Formatter, StreamHandler, getLogger

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
