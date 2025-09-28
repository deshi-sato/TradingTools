# scripts/fetch_ranking.legacy.py 仕様書

## 概要
- kabuステーション REST `/ranking` API から出来高・売買代金・値上がり/値下がりランキングを収集し、`perma_regulars.csv` を生成する旧版クライアント。
- カテゴリごとに最大件数を切り替えながら集計し、銘柄ごとに理由タグを重複なくまとめる。

## 主な機能
- `CATEGORY_MAP` / `EXDIV_MAP` を用いてAPIパラメータを日本株市場コードへ変換。
- APIレスポンスの揺らぎ（大文字小文字、フィールド名差異）に対応する `normalize_item()` を実装。
- 取得件数をカテゴリ別に制御し、理由タグ（`turnover`,`volume`,`up`,`down` など）をソートしてCSV化。
- `--dry-run` で標準出力に結果を流し、`--debug` で詳細ログを有効化。

## 主な引数
| 引数 | 説明 |
|------|------|
| `-Config` | 必須。`kabu.base_url` / `kabu.api_token` / `ranking.market_code` 等を含むJSON。|
| `--limit` | 全カテゴリ共通の上限件数を上書き。未指定時はカテゴリ既定値。|
| `--output` | 出力CSVパス。未指定時は設定ファイルの `default_output`。|
| `--dry-run` | ファイル書き込みを行わず、結果を標準出力にCSV形式で表示。|
| `--debug` | ログレベルをDEBUGに変更。|

## 設定項目
| キー | 説明 |
|------|------|
| `kabu.base_url` | kabuステーションAPIのベースURL。末尾スラッシュは自動調整。|
| `kabu.api_token` | `/ranking` 呼び出しに使用するAPIトークン。|
| `ranking.market_code` | 取引所セグメントを指定（1=全市場、101=東証プライムなど）。|
| `ranking.default_output` | CSV出力先の既定値。|
| `ranking.timeouts_sec` | HTTPタイムアウト秒。|

## 処理フロー
1. 設定JSONを読み込み、APIベースURL・トークン・市場コードなどを取得。
2. `per_category_limits` を決定し、`aggregate_rankings()` でカテゴリごとに /ranking API を呼び出す。
3. 各カテゴリ応答を `normalize_item()` で整形し、銘柄ごとに理由タグ集合を統合。
4. `--dry-run` の場合は `print_csv()` で即座に標準出力へ出力。
5. 通常時はUTF-8(BOM)でCSVを書き出し、成功/失敗をログへ記録。

## 入出力
- 入力: kabuステーション `/kabusapi/ranking` エンドポイント。
- 出力: `code,name,reason` の3列CSV（`perma_regulars.csv`など）。
- ログ: 進行とHTTPエラーを標準出力/標準エラーに記録。

## 依存
- `requests` ライブラリを利用（存在しない環境では None を代入し静的解析を回避）。
- `scripts/common_config.load_json_utf8` による設定読込。

## 実行例
```powershell
py scripts/fetch_ranking.legacy.py -Config config/stream_settings.json --dry-run
py scripts/fetch_ranking.legacy.py -Config config/stream_settings.json --output data/perma_regulars.csv
```
