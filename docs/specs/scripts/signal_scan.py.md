# scripts/signal_scan.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 args.dailydb
- DB接続: sqlite3 args.minutedb
- DB接続: sqlite3 args.newsdb
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- JST_DATEFMT: '%Y-%m-%d'
- JST_DATETIMEFMT: '%Y-%m-%d %H:%M:%S'
- GAP_TH: 0.005
- BRK_EPS: 0.0005
- VOL_SPIKE_K: 3.0
- MA_WINDOW: 20
- TRADING_SESSIONS: [
    (time(9, 0), time(11, 30)),
    (time(12, 30), time(15, 25)),
]
- CATEGORY_WEIGHTS: {
    "材料": 2.0, "特報": 2.0, "決算": 1.6, "注目": 1.0,
    "市況": 0.5, "テク": 0.6, "速報": 1.0, "通貨": 0.5,
    "経済": 0.5, "業界": 0.6, "特集": 0.5, "総合": 0.4, "５％": 0.5,
}
- POS_WORDS: ["上方修正","増配","最高益","上振れ","通期上方","自社株買い","大型受注","好調","過去最高"]
- NEG_WORDS: ["下方修正","減配","赤字","下振れ","不適切会計","監理","公募増資","売出","未達","据え置き"]
- CONS_POS: ["市場予想を上回る","コンセンサス超え","予想を上回る","上振れ","ガイダンス上方"]
- CONS_NEG: ["市場予想を下回る","コンセンサス未達","予想を下回る","下振れ","据え置き","ガイダンス下方"]
- STRONG_POS: ["大幅に上回る","大きく上回る","サプライズ上振れ"]
- STRONG_NEG: ["大幅に下回る","大きく下回る","サプライズ下振れ"]

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- ticker_candidates(ticker: str) -> None: '3382.T' でも '3382' でもヒットするよう候補を返す
- in_sessions(t: time) -> bool: 説明なし
- _norm_code4(ticker: str) -> str: '7453.T' -> '7453' / 先頭4桁を抽出
- _to_yyyymmdd(date_str: str) -> str: 説明なし
- _pick_hhmm(x: str) -> str | None: 混在フォーマットから HH:MM を抜き出す（例: '25/08/25 05:35' -> '05:35'）
- consensus_hint_from_text(text: str) -> int: ニュース本文/タイトルからコンセンサス表現を±2〜±1で返す（1記事単位）
- load_news_features_kabutan(conn_news, ticker: str, date_str: str) -> dict: 当日(YYYYMMDD) + 前日(15:00-23:59) をスコア化して合算。
- load_watchlist(path_watchlist: Path) -> pd.DataFrame: 説明なし
- load_prev_ohlc(conn_daily, ticker, date_str) -> None: 説明なし
- load_intraday(conn_min, ticker, date_str) -> None: 説明なし
- detect_signals_for_ticker(ticker, side, date_str, conn_daily, conn_min, conn_news = None) -> None: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- なし

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
