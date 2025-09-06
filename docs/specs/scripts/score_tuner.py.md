# scripts/score_tuner.py 仕様書

## 概要
score_tuner.py  (daily-ready, summary + per-code detail exporter)

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 db_path
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- なし

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- fetch_minutes_as_df(db_path: str, table: str, date_col: str, code_col: str, price_col: str, volume_col: str, start: str, end: str, pad_days: int = 5) -> pd.DataFrame: SQLiteから必要列を取得（dt, code, close, volume, high）。日足テーブルでもOK。
- minutes_to_daily(df: pd.DataFrame) -> pd.DataFrame: 標準列(dt, code, close, volume, high) → 日足(date, code, close, volume, high)
- build_prev_next_pairs(daily: pd.DataFrame) -> pd.DataFrame: 各銘柄ごとに Prev→Next ペアを作る（Prevでスコア、Nextでリターン）。
- compute_factor_scores(paired: pd.DataFrame) -> pd.DataFrame: 3因子:
- evaluate_by_date(df_scores: pd.DataFrame, w: Tuple[float, float, float], dates: List[str], rank_method: str = 'spearman', topn: int | None = None) -> Tuple[float, float, int]: 説明なし
- export_codes_detail(scored: pd.DataFrame, w: Tuple[float, float, float], use_dates: List[str], out_base: str) -> str: 評価に使った全日付について [date,code,score,next_return] を保存。
- main() -> None: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
