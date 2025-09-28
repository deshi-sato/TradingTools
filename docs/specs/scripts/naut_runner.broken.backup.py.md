# scripts/naut_runner.broken.backup.py 仕様書

## 概要
- `naut_runner.py` をPowerShell経由で差し替えるための手順書兼スクリプト。`Copy-Item` とヒアドキュメントを用いて、修正版ソースを `scripts/naut_runner.py` に書き戻す。
- Pythonコードとして実行することは想定されておらず、リポジトリ内に修正前後の比較記録を残す目的で保管されている。

## 主な内容
- 手順1で現行 `naut_runner.py` を `naut_runner.broken.backup.py` にコピー（バックアップ取得）。
- 手順2で `@' ... '@` ヒアドキュメントに修正版Pythonコードを埋め込み、`Set-Content` で上書きするPowerShellスクリプトを記述。
- ヒアドキュメント内には修正版の `naut_runner.py` 全文が含まれており、PowerShellからそのまま実行すると再適用できる。

## 利用手順
1. PowerShellを開き、リポジトリルートで `.\scripts\naut_runner.broken.backup.py` の内容を確認する。
2. 手順コメントに従って `Copy-Item` と `Set-Content` を順に実行し、修正版 `naut_runner.py` を再生成する。
3. 必要に応じてヒアドキュメント中のソースを編集してから適用する。

## 注意点
- Pythonとして実行するとエラーになるため、必ずPowerShellスクリプトとして扱う。
- ファイル名に `.py` が付いているが、静的解析やツールチェーンから除外する必要がある場合は個別に設定する。
- 修正版コードは `scripts/naut_runner.py` と同一内容であり、差分検証後は本ファイルを参照しなくてもよい。

## 補足
- 最新の運用コードは `scripts/naut_runner.py` にあり、本ファイルは緊急時の復旧用バックアップとして維持されている。
