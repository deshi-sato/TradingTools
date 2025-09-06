# scripts/push_to_notion.py 仕様書

## 概要
Post a summary entry to a Notion database with Japanese properties:

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: req
- コマンドライン引数: argparse によるオプションを受け付けます

## 出力
- なし

## 設定項目
- NOTION_API: 'https://api.notion.com/v1'
- NOTION_VERSION: '2022-06-28'

## 処理フロー
- 起動: __main__ ブロックあり
- 引数解析: argparse でオプションを解析
- 入出力: ファイルの読み書きを実施
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- parse_args() -> argparse.Namespace: 説明なし
- notion_headers(token: str) -> Dict[str, str]: 説明なし
- notion_request(method: str, path: str, token: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]: 説明なし
- get_db_schema(token: str, db_id: str) -> Dict[str, Any]: 説明なし
- find_title_prop_name(db_json: Dict[str, Any]) -> str: 説明なし
- to_rich_text(text: str) -> Dict[str, Any]: 説明なし
- resolve_child_database(token: str, page_id: str) -> Optional[str]: 説明なし
- main() -> None: 説明なし

## 代表的なエラー
- Exception
- error.HTTPError

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
