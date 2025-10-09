# DeepNaut 開発仕様

- [[deepnaut.dev.daily.2025-10-09]]
- [[deepnaut.ops.chat.2025-10-09]]
---
title: DeepNaut 開発仕様
---

# DeepNaut 開発仕様（概要）

## システム構成
- kabuステーションAPI（PUSH + REST）
- stream_microbatch：PUSH受信・特徴量算出
- naut_runner：シグナル検知・発注制御
- rss_snapshot.db：ティック・板データ格納
- fallback_scraper：バックアップ取得

## 運用ルール
- 稼働時間：9:00〜15:30
- BUY閾値：0.6 / SELL閾値：0.4（暫定）
- GridSearch により随時調整
- OCO決済を原則、成行は例外的使用
- 前景実行による安定検証を優先

## 主要ノート
- [[deepnaut.dev.design]]
- [[deepnaut.dev.impl.stream_microbatch]]
- [[deepnaut.dev.daily.2025-10-09]]
- [[deepnaut.ops.chat.2025-10-09]]
- [[deepnaut.ops.todo]]
- [[deepnaut.ops.log.changes]]
