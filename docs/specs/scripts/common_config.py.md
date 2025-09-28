# scripts/common_config.py 仕様書

## 概要
- 設定ファイル（JSON）をUTF-8(BOM対応)で読み込み、辞書として返す汎用ヘルパー。
- 文字化けやJSON構造の破損を検出し、再利用しやすいメッセージで `RuntimeError` を投げる。

## 主な機能
- `_snippet_from_bytes()` により読み込み失敗時の先頭断片を取得し、エラーメッセージに埋め込む。
- BOM付きUTF-8を優先して読み込み、`UnicodeDecodeError` を捕捉して詳細な原因を付与。
- `json.loads()` でデコードした結果がオブジェクト（dict）であることを保証。

## 関数API
| 関数 | 説明 |
|------|------|
| `_format_snippet(text: str)` | 改行を空白に置換し、先頭120文字をサマリとして返す。|
| `_snippet_from_bytes(path: Path)` | バイナリ読み込み（最大480バイト）からUTF-8デコードし、エラースニペットを生成。|
| `load_json_utf8(path: str)` | UTF-8(BOM対応)でJSONを読み込み、辞書を返す。失敗時は `RuntimeError`。|

## 処理フロー
1. 対象パスを`Path`化し、UTF-8(BOM)でテキストを読み込む。失敗時は `_snippet_from_bytes` を使ってエラー補足。
2. JSONパースを行い、`json.JSONDecodeError` が発生した場合は生テキストの冒頭を `_format_snippet` で添えて報告。
3. パース結果が辞書でなければ型情報を含む `RuntimeError` を送出。
4. 正常時は辞書を返却し、設定読込の標準ルートとして利用。

## 連携
- `scripts/fetch_ranking.py` など多くのCLIスクリプトが設定読込で共通利用。
- 例外メッセージがユーザー向けに整形されているため、上位層でそのまま表示するだけで原因把握が容易。

## 実行例
```python
from scripts.common_config import load_json_utf8
cfg = load_json_utf8('config/stream_settings.json')
print(cfg['db_path'])
```
