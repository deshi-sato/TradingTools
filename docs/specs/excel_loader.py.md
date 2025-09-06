# excel_loader.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: src_path
- ファイル入力: dst_path

## 出力
- なし

## 設定項目
- EXCEL_PATH_L: 'C:/Users/Owner/Documents/desshi_signal_viewer/買い銘柄寄り後情報.xlsm'
- EXCEL_PATH_S: 'C:/Users/Owner/Documents/desshi_signal_viewer/売り銘柄寄り後情報.xlsm'
- RSS_PARAM_TO_REPLACE: '1660'
- RSS_PARAM_NEW: '332'
- SCORE_THRESHOLD_L: 7
- SCORE_THRESHOLD_S: 4

## 処理フロー
- 起動: __main__ ブロックあり
- 入出力: ファイルの読み書きを実施
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- get_latest_date_from_data(file_path) -> None: Excelファイルから最新の日付を取得する
- load_summary_data(file_path) -> None: 説明なし
- load_data(file_path) -> None: 説明なし
- export_sheets(src_path, top_long, top_short, code_to_name) -> None: 説明なし
- export_top_sheets() -> None: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
