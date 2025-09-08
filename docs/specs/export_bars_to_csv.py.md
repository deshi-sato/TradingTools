# export_bars_to_csv.py 仕様書

## 概要
SQLiteの任意テーブルから、timestamp,open,high,low,close,volume,symbol 列でCSVを書き出す汎用ツール。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 args.db
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
- ensure_iso8601(ts: object, is_date: bool = False) -> str: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- Exception
- KeyboardInterrupt
- sqlite3.OperationalError

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
