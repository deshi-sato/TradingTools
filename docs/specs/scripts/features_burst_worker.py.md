# scripts/features_burst_worker.py 仕様書

## 概要
- `features_stream` テーブルをポーリングし、バースト系フラグ（`burst_buy`/`burst_sell`）やスコア列をリアルタイムに埋めるワーカー。
- 直近ウィンドウのローリング統計とEMAを組み合わせ、勢いの継続性やボリューム急増を判定する。

## 主な機能
- JSON設定の `burst` セクションから閾値・ウィンドウ長・クールダウン秒などを読み取り、判定ロジックを柔軟に調整。
- `features_stream` の新規行だけを `rowid` 増分で取得し、データ無し時はスリープする軽量ポーリングを実現。
- uptick比率、板厚、ボリューム合計からバースト買い／売りを算出し、指数移動平均を用いたボリュームサージ比でスコアリング。
- 連続発火防止のためシンボル単位でクールダウン秒とEMAストリークを追跡し、`surge_vol_ratio`/`streak_len` などの列を更新。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`db_path` と `burst` 設定を含むJSON。|
| `-Verbose` | ログ閾値（1=INFO, 0=WARNING）。|

## 設定パラメータ（例）
| キー | 説明 |
|------|------|
| `burst.window_count` | 判定ウィンドウ内の行数 (`K`)。|
| `burst.uptick_thr_buy` / `burst.uptick_thr_sell` | 買い/売り判定で必要なuptick比率。|
| `burst.imb_thr` | 板厚アンバランスの閾値。|
| `burst.max_spread_bp` | スプレッド上限（bp）。|
| `burst.vol_gate` / `burst.vol_sum_gate` | 直近行およびウィンドウ合計のボリューム下限。|
| `burst.allow_spread_none_with_vol` | スプレッド不明でも合計ボリュームが閾値超なら許容するか。|
| `burst.cooldown_sec` | シンボルごとの再発火待機秒数。|
| `burst.ema_span_sec` | ボリュームEMAの時定数。|
| `burst.burst_score_weights` | uptick / imbalance / vol_surge 各要素の重み。|

## 処理フロー
1. 設定読込後、SQLite接続を確立し `rowid` > `last_rowid` の新規行を取得するループへ入る。
2. シンボルごとにデックへ最新 `K` 件の `(uptick, vol, spread, imbalance)` を蓄積し、EMAで平滑化した平均ボリュームと比較してサージ比を算出。
3. スプレッド許容・ボリューム閾値・uptick投票・アンバランス投票を評価し、買い/売りの一貫性が成立すれば `burst_buy` / `burst_sell` を立てる。
4. 指定ウェイトで `burst_score` を計算し、クールダウン中は強制的にフラグを落とす。
5. ストリーク長やサージ比、最終シグナル日時を組み合わせた更新パラメータを `UPDATE features_stream` へ一括書き込み。
6. 追加行が無い場合は 150ms スリープして再ポーリング。

## 更新列
- `burst_buy`, `burst_sell`, `burst_score`, `streak_len`, `surge_vol_ratio`, `last_signal_ts` を`rowid`に基づき更新。

## 連携
- `scripts/naut_runner.py` シリーズが参照するバースト指標をリアルタイム補完。
- `scripts/burst_helper.py` の `burst_bonus` と組み合わせることで二次判定に利用。

## 実行例
```powershell
py scripts/features_burst_worker.py -Config config/stream_settings.json -Verbose 1
```
