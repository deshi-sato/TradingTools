# delete_after_1130.py 仕様書

## 概要
Delete today's minute_data rows at/after 11:30 (local time).

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- DB接続: sqlite3 str(db_path)
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
- build_args() -> argparse.Namespace: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- なし

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
