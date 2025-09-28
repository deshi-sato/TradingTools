# scripts/burst_helper.py 仕様書

## 概要
- `features_stream` テーブルから直近のバーストシグナルを集計し、戦略スコアにボーナスを付加するユーティリティ関数群。
- バースト判定の寿命や冷却時間を制御し、`naut_runner` などの約定判定ロジックに再利用できるよう設計されている。

## 主な機能
- `fetch_recent_burst_strength()` で指定ティッカーの直近シグナルを走査し、スコアとタイムスタンプを返却。
- `burst_bonus()` で基礎スコアに指数減衰を掛けたボーナスを合成し、冷却済みかどうかを返す。
- ISO8601文字列をUTC naïve datetimeへ変換する `_to_dt()` を内包し、時刻比較を安定化。

## 関数API
| 関数 | 説明 |
|------|------|
| `_to_dt(ts: str)` | ISO8601文字列をUTCのnaïve `datetime` に変換。失敗時は `None`。|
| `fetch_recent_burst_strength(conn, ticker, within_sec, min_gate)` | `features_stream` から最新50件を参照し、スコア閾値と経過秒数でフィルタした最大スコア・時刻を返す。|
| `burst_bonus(conn, ticker, base_score, k, tau_sec, lookback_sec, min_gate)` | 直近バーストに指数減衰ボーナスを掛け、更新後スコア・ボーナス値・シグナル発生時刻を返す。|

## 処理フロー
1. `fetch_recent_burst_strength` はティッカー・フラグ列 (`burst_buy`/`burst_sell`) で絞り込み、最新シグナルから順に時刻やスコアを検証する。
2. `within_sec` が設定されていれば現在時刻との差分を計算し、許容時間を超える行をスキップ。
3. ボーナス計算では `age = now - ts` を指数関数で減衰させ、`bonus = k * score * exp(-age / tau_sec)` を適用。
4. シグナルが見つからない場合は `(base_score, 0.0, None)` を返し、後段で冷却判定しやすくする。

## 連携
- `scripts/naut_runner.py` / `scripts/replay_naut.py` でトレード判定時に呼び出し、爆発的なボリューム継続を優遇する。
- SQLiteコネクションを共有するため、呼び出し元でトランザクション管理を行う。

## 実行例
```python
import sqlite3
from scripts.burst_helper import burst_bonus

conn = sqlite3.connect('data/rss_snapshot.db')
score, bonus, ts = burst_bonus(conn, '7203', base_score=0.55, k=0.3, tau_sec=12.0)
print(score, bonus, ts)
```
