# scripts/replay_naut.py 仕様書

## 概要
- `naut_runner` 系ロジックをリプレイ用途向けに切り出したスクリプト。現行実装は `scripts/naut_runner.py` と同等の処理を含み、過去データに対する検証や手動デバッグで利用する前提で保管されている。

## 主な機能
- BUY/SELL判定やバーストボーナス計算など、`scripts/naut_runner.py` と同じヘルパー (`burst_helper`, `common_config`) を利用してシグナルを評価。
- プロファイル単位で閾値セットを切り替え、採用・棄却統計を `--print-summary` で確認できる。
- SQLite接続やストリーク管理などの挙動も本番ロジックと同一のため、設定ファイルを切り替えるだけで過去データを再生できる。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`settings.naut` と `db_path` を含むJSON。|
| `-Verbose` | ログレベル（1=INFO、0=WARNING）。|
| `--profile` | `prod` / `prodlite` / `demo` から閾値セットを選択。|
| `--print-summary` | 1サイクルのみ実行し、統計情報を出力して終了。|

## 利用シナリオ
- 過去の `features_stream` スナップショットDBを指定してシグナル判定の再現性を確認する。
- 本番ロジックに変更を加える前にリプレイ用コピーとして差分検証する。

## 注意点
- 現在のコードは `scripts/naut_runner.py` のコピーであり、今後リプレイ専用機能を分離する際のベースとして維持されている。
- 実行環境（設定・DB）は `naut_runner` と共通で、同時起動時は同じDBロックに注意。

## 実行例
```powershell
py scripts/replay_naut.py -Config config/stream_settings.json --profile prod --print-summary
```
