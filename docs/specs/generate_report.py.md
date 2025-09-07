# generate_report.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: <動的パス>
- ファイル入力: <動的パス>
- ファイル入力: 'r'
- ファイル入力: <動的パス>
- ファイル入力: <動的パス>
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- STRATEGY_KEYS: {"strategy", "name", "model", "signal", "label", "tag"}
- SHARPE_ALIASES: {"sharpe", "sharperatio", "sharpe_ratio"}
- CUM_ALIASES: {
    "cumulativereturn",
    "cumulative",
    "cumreturn",
    "totalreturn",
    "returncumulative",
    "cumend",
}
- MAXDD_ALIASES: {"maxdd", "maxdrawdown", "maxdrawdownratio"}

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- 入出力: ファイルの読み書きを実施
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- normalize_key(s: str) -> str: 説明なし
- parse_float(s: str) -> Optional[float]: 説明なし
- _sniff_dialect(path: Path) -> Optional[_csv.Dialect]: 説明なし
- read_summary(path: Path) -> Dict[str, Metrics]: 説明なし
- read_monthly_table(path: Path) -> Tuple[List[str], Dict[str, List[float]], Dict[str, Optional[float]], Dict[str, Optional[float]]]: 説明なし
- mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]: 説明なし
- compute_sharpe(monthly: List[float], periods_per_year: int = 12) -> Optional[float]: 説明なし
- compute_cum(monthly: List[float]) -> Optional[float]: 説明なし
- compute_maxdd(monthly: List[float]) -> Optional[float]: 説明なし
- pct(x: Optional[float], digits: int = 1, sign: bool = False) -> str: 説明なし
- fmt_float(x: Optional[float], digits: int = 2) -> str: 説明なし
- build_report(summary_csv: Path, monthly_csv: Path, images: List[Path], out_md: Path, title: str) -> None: 説明なし
- main() -> None: 説明なし
- Metrics: 説明なし

## 代表的なエラー
- Exception
- ValueError

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
