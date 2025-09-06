# scripts/fix_excel_formulas.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- なし

## 出力
- なし

## 設定項目
- TARGET_CELLS: {"A1", "AA1"}
- SUFFIX_RE: re.compile(r"[\s]*=>\s*関数使用上限に達しました。?$")

## 処理フロー
- 起動: __main__ ブロックあり
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- _ns(tag) -> None: 説明なし
- clean_sheet_xml(data: bytes) -> bytes: 説明なし
- process_file(path: Path) -> None: 説明なし
- main(argv) -> None: 説明なし

## 代表的なエラー
- ET.ParseError

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
