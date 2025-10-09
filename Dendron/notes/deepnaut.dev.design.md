---
title: 設計思想
---

# DeepNaut 設計思想

## 基本方針
- 「小さな勝ちを積み上げる」短期順張りロジック。
- データストリームを秒単位で評価し、BUY/SELL判定を閾値制御。
- 複雑な予測モデルよりも **速度と透明性** を重視。

## 特徴量設計
- ティック変化率・出来高・板気配・VWAP乖離などを使用。
- Stream層では特徴量を秒単位で更新。
- Replay層では日次再評価を実施。

## アーキテクチャ
fallback_scraper → rss_snapshot.db
↓
stream_microbatch (前景)
↓
naut_runner → 発注処理(OCO)

markdown
コードをコピーする

## 今後の拡張
- SELL閾値をGridSearch自動最適化に統合
- 特徴量をMLモデルに渡す仕組みを検討
