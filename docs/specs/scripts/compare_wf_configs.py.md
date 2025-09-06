# scripts/compare_wf_configs.py 仕様書

## 概要
Compare walk-forward daily results for selected tags (e.g., T60_V30,T90_V30).

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- 環境変数: name

## 出力
- なし

## 設定項目
- OUTDIR: os.path.join("data", "analysis")

## 処理フロー
- 起動: __main__ ブロックあり
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- read_env_list(name: str) -> List[str]: 説明なし
- read_env_int(name: str, default: int) -> int: 説明なし
- find_available_tags() -> List[str]: 説明なし
- _safe_tag(tag: str) -> str: 説明なし
- pick_tags() -> List[str]: 説明なし
- max_drawdown(cum_series: pd.Series) -> float: 説明なし
- load_daily(tag: str) -> pd.DataFrame: 説明なし
- make_cum(ret: pd.Series) -> pd.Series: 説明なし
- monthly_agg(df: pd.DataFrame) -> pd.DataFrame: 説明なし
- main() -> int: 説明なし

## 代表的なエラー
- Exception
- ValueError

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
