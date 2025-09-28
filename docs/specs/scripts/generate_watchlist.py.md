# scripts/generate_watchlist.py 仕様書

## 概要
- 取引ユニバース/日次ランキング/手動人気リストを統合し、スコアリングした上で当日のウォッチリストを生成するメインロジック。
- JSON設定で重み・閾値・ファイルパスを柔軟に調整し、リブートなしで日次運用に対応。

## 主な機能
- `load_config()` で設定ファイルとデフォルト値をマージし、重みや閾値、ランダムシード等を構造体 (`Config`) にまとめる。
- ユニバースCSV／perma_regulars／manual_popular を読み込み、正規表現ベースのコード抽出や重複除去を行う。
- 各ソースに応じたスコア貢献（vol_surge、turnover、news_pop、manual/permaボーナス等）を算出し、閾値に満たないものを切り捨て。
- ランダムシードを固定しながらスコアが同値の銘柄の順序を安定化。上位N件を `watchlist_today` と `watchlist_top50` に書き出す。
- `--dry-run` 時はテーブル形式のプレビューを標準出力へ表示し、ファイル出力を抑制。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 設定JSON。未指定時は `config/stream_settings.json` を参照。|
| `--max` | 出力件数の上限を上書き。|
| `--seed` | 重複解消時に用いる乱数シードを上書き。|
| `--output` | メイン出力CSVパスを上書き。|
| `--dry-run` | ファイル書き出しの代わりにプレビュー表を表示。|
| `--debug` | ログレベルをDEBUGに変更。|
| `--preview` | ドライラン時のプレビュー件数。既定10。|

## 設定項目（抜粋）
| セクション | キー | 説明 |
|-----------|------|------|
| `paths` | `universe`, `perma_regulars`, `manual_popular`, `output` | 入力/出力ファイルの既定パス。|
| `limits` | `max_output` | 出力上限件数。|
| `weights` | `vol_surge`, `turnover`, `depth_stable`, `news_pop`, `manual_bonus`, `perma_bonus` | 各スコア要素の重み。|
| `thresholds` | `vol_surge`, `turnover`, `depth_stable`, `news_pop` | 採用判定の下限値。|
| `format` | `score_decimals` | スコア出力時の小数桁数。|

## 処理フロー
1. 設定を読み込み、必要に応じてCLIオプションで上書き。ログレベルを初期化。
2. ユニバース・perma・manualソースを順に読み込み、コードと名称を正規化しながら重複管理。
3. 各銘柄に対して重み付きスコアを計算し、閾値を満たさない要素をスキップ。ボーナスを合算して総合スコアを得る。
4. 重複を排除した上でスコア降順にソートし、`max_output` 件を採用。順位 (`rank`) を付与。
5. `--dry-run` の場合は整形テーブルを表示して終了。通常時はメインリストとTop50リストのCSVを書き出す。
6. 処理サマリ（読み込んだ件数・有効件数・重複数・出力件数・経過時間）をINFOログへ出力。

## 入出力
- 入力: 複数CSV (`universe.csv`, `perma_regulars.csv`, `manual_popular.csv`) と設定JSON。
- 出力: `watchlist_today.csv` と `watchlist_top50.csv`（UTF-8, `rank,code,name,score,Reason`）。
- ログ: 標準出力にINFO/DEBUGを出力。

## 実行例
```powershell
py scripts/generate_watchlist.py -Config config/stream_settings.json --max 70
py scripts/generate_watchlist.py -Config config/stream_settings.json --dry-run --preview 20 --debug
```
