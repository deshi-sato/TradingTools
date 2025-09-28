# scripts/naut_runner.backup.py 仕様書

## 概要
- `scripts/naut_runner.py` の旧実装をバックアップとして保管したファイル。基本的な判定ロジックやCLI構成は現行版と同等で、解析や比較検証に利用する。
- バースト判定・ストリーク管理・閾値プロファイルなど、かつてのNaut実行フローをそのまま残している。

## 主な機能
- `compute_base_score()` でuptick比率と板アンバランス、出来高超過を組み合わせた基礎スコアを算出。
- `burst_helper.burst_bonus()` を適用して直近シグナルの勢いを加点、クールダウンや連続判定の可否を判断。
- プロファイル設定（`prod`/`prodlite`/`demo`）を切り替え、買い/売りそれぞれの閾値やウィンドウ長を調整。
- `--print-summary` 指定時にはループを1周だけ走らせ、採用・スキップ理由の統計を標準出力へダンプ。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`settings.naut` やDBパスを含むJSON設定。|
| `-Verbose` | ログレベル（1=INFO, 0=WARNING）。|
| `--print-summary` | ループを1回で止め、判定サマリを出力。検証用途。|
| `--profile` | `prod` / `prodlite` / `demo` から閾値セットを選択。|

## 処理フロー
1. 設定ファイルから `naut` セクションを読み取り、プロファイル別の閾値（uptick門番、spread上限、vol_minなど）を適用。
2. `features_stream` から最新のフィーチャーを取得し、シンボルごとにローリング統計を管理。
3. 買い/売り候補に対して一貫性（streak）やバーストボーナス、冷却時間を判定し、採用されたシグナルをログ出力。
4. `--print-summary` が有効な場合は1サイクル後に判定数と棄却理由の内訳を表示して終了。通常時は0.25秒周期でループ継続。

## 連携
- 参照DBや配置は現行 `scripts/naut_runner.py` と共有。差分検証やリファレンスとして利用する。
- `scripts/replay_naut.py` などのリプレイ系スクリプトの原型となっている。

## 実行例
```powershell
py scripts/naut_runner.backup.py -Config config/stream_settings.json --profile prodlite --print-summary
```

> ⚠️ 本ファイルはバックアップ目的のため、運用には現行の `scripts/naut_runner.py` を使用すること。
