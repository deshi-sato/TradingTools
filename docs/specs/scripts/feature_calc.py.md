# scripts/feature_calc.py 仕様書

## 概要
- 板情報やティック統計から特徴量を算出する純粋関数群。`stream_microbatch` や `stream_microbatch_rest` で再利用される。
- 上位板数量合計・スプレッド・板厚アンバランス・uptick比率など、`features_stream` 列に直接対応する基本指標を提供。

## 関数API
| 関数 | 説明 |
|------|------|
| `top3_sum(levels)` | 板の上位3本分の数量合計を整数で返す。空やNoneの場合は0。|
| `spread_bp(bid1, ask1)` | 中値を基準にスプレッドをbp換算で返す。片側欠損や0除算は `None`。|
| `depth_imbalance(buy_top3, sell_top3)` | 上位買い／売り厚の差を正規化して-1〜+1範囲の指標を返す。|
| `uptick_ratio(upticks, downticks)` | uptick件数の比率 (`upticks/(upticks+downticks)`) を返す。件数ゼロ時は0。|

## 利用上の注意
- 入力はPython組込み型（list/tuple/float/int）で、外部依存は無い。
- `spread_bp` はミッドプライス<=0の場合に `None` を返すため、呼び出し側でカバー値を決めること。

## 利用例
```python
from scripts.feature_calc import top3_sum, spread_bp, depth_imbalance

bids = [(300.0, 1200), (299.9, 800), (299.8, 600)]
asks = [(300.2, 900), (300.3, 700)]
print(top3_sum(bids))            # -> 2600
print(spread_bp(299.9, 300.2))   # -> 100.1bp
print(depth_imbalance(2600, 1600))
```
