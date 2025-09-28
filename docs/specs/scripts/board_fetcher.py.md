# scripts/board_fetcher.py 仕様書

## 概要
- kabuステーションの板情報をRESTポーリングで取得するための簡易フェッチャー。
- PUSHが使えない環境でも最低限の板スナップショットを生成する開発用スタブ実装。

## 主な機能
- `mode`の指定に応じてREST専用・PUSH専用・自動切り替え（初回のみREST）を想定した動作を提供。
- シンボルごとに直近のRESTアクセス時刻を `_LAST_REST_TS` に保持し、`rest_poll_ms` 間隔を下回る呼び出しをフォールバック扱いにする。
- REST経由で取得できた場合は`bid1`/`ask1`と上位3本の板を含む辞書を返し、保護のために固定値を返すスタブ構造を維持。
- 規定間隔内で呼び出された場合は空の板（`bid1`/`ask1`なし）を返してダミーのフォールバック分岐をテスト可能にする。

## 主な引数
| 引数 | 説明 |
|------|------|
| `mode` | `"auto"` / `"push"` / `"rest"` を想定した動作モード。スタブでは`auto`時にRESTへフォールバックする流れを再現。|
| `rest_poll_ms` | 同一シンボルに対するRESTアクセスの最小間隔（ミリ秒）。既定値500。|

## 処理フロー
1. `get_board(symbol)` を呼ぶとモノトニックタイマーから現在時刻を算出し、最後にRESTを叩いた時刻と比較する。
2. `mode`が`auto`または`rest`で、かつ待機間隔を超えていればRESTを想定した固定データ（`bid1`/`ask1`・`bids`・`asks`）を返す。
3. インターバル内で呼ばれた場合はフォールバックに入り、板が取得できなかった前提で空の辞書を返す。
4. PUSH経路を再開したいときは`start_push()`（現在はスタブ）を拡張し、実際のWebSocket購読処理に差し替えることを想定。

## 連携・依存
- `scripts/stream_microbatch.py` や `scripts/stream_microbatch_rest.py` の板取得ロジック差し替えテストに利用する前提のスタブ。
- ロギングは`logging.getLogger(__name__)`を使用するため、呼び出し側でログレベルを制御可能。

## 実行例
```python
from scripts.board_fetcher import BoardFetcher

fetcher = BoardFetcher(mode="rest", rest_poll_ms=500)
board = fetcher.get_board("7203")
print(board["bid1"], board["asks"][:2])
```
