# scripts/check_db.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: CODES_TXT
- DB接続: sqlite3 DB

## 出力
- ファイル出力: out_csv mode='w'
- ファイル出力: miss_path mode='w'

## 設定項目
- DB: 'data/rss_daily.db'
- CODES_TXT: 'data/topix100_codes.txt'

## 処理フロー
- 起動: __main__ ブロックあり
- 入出力: ファイルの読み書きを実施
- データアクセス: sqlite3 に接続・操作
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- q(conn, sql, params = ()) -> None: 説明なし
- print_rows(title, cols, rows, limit = 10) -> None: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- なし

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
