---
title: stream_microbatch 実装メモ
---

# stream_microbatch.py 実装メモ

## 目的
PUSH配信から秒単位の特徴量を生成し、DBへ保存する。

## 主な更新点（2025-10-09）
- BUY閾値に新設定値（0.6）を反映。
- SELL閾値は既定値（0.4）で保留。
- 前景モード起動（Start-microbatch.ps1）で動作安定を確認。

## 実行順序
1. kabus_login_wait.py
2. build_fallback_scraper
3. watchlist 登録
4. stream_microbatch（前景起動）

## ログ
- 出力：`naut_live_2025-10-09.log`
- BUY判定：閾値反映済み
- SELL判定：次回検証予定

## 次回改善
- GridSearchに基づくSELL閾値反映
- PUSH受信例外処理の強化
