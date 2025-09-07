# tools/auto_spec.py 仕様書

## 概要
このスクリプトの高レベルな機能を記述してください。

## 目的
業務/運用上の目的を簡潔に記述してください。

## 入力
- ファイル入力: path

## 出力
- ファイル出力: path mode='w'

## 設定項目
- EXCLUDE_DIRS: {"tests", "test", ".git", ".github", "venv", ".venv", "__pycache__"}

## 処理フロー
- 起動: __main__ ブロックあり
- 入出力: ファイルの読み書きを実施
- コア処理: 主要関数を順次呼び出し

## 主要関数・クラス
- read_text_best_effort(path: str) -> str: 説明なし
- list_python_files(root: str) -> List[str]: 説明なし
- get_docstring(node: ast.AST) -> Optional[str]: 説明なし
- name_of(node: ast.AST) -> str: 説明なし
- const_str(node: Optional[ast.AST], source: str) -> Optional[str]: 説明なし
- format_args_sig(args: ast.arguments, source: str) -> str: 説明なし
- collect_symbols(tree: ast.Module, source: str) -> Dict[str, Any]: 説明なし
- render_markdown(rel_path: str, source: str, info: Dict[str, Any]) -> str: 説明なし
- write_file(path: str, content: str) -> None: 説明なし
- rel_to_spec_path(rel_path: str) -> str: 説明なし
- main(argv: List[str]) -> int: 説明なし

## 代表的なエラー
- Exception

## ログ
- なし

## 注意点・制約
- 実行環境: Python 3.x 標準ライブラリで動作
- パフォーマンス: 入出力/DBアクセス量によって変動
- 前提: 必要な入力ファイル/DBが存在すること
