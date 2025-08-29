# -*- coding: utf-8 -*-
import re
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook

DATA_DIR = Path("./data/analysis")
SNAPSHOT_XLSM = Path("rss_snapshot.xlsm")
SHEET_NAME = "index"
A_MAX = 30

# ファイル名から日付を取る用: watchlist_YYYY-MM-DD.csv
WATCH_RE = re.compile(r"watchlist_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)


def pick_latest_watchlist() -> Path:
    candidates = list(DATA_DIR.glob("watchlist_*.csv"))
    if not candidates:
        raise FileNotFoundError("data フォルダに watchlist_*.csv が見つかりません。")

    # 1) ファイル名に日付があればそれで最大日付を優先
    dated = []
    others = []
    for p in candidates:
        m = WATCH_RE.search(p.name)
        if m:
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                dated.append((d, p))
            except ValueError:
                others.append(p)
        else:
            others.append(p)
    if dated:
        dated.sort(key=lambda x: x[0])
        return dated[-1][1]
    # 2) それ以外は更新時刻が新しいもの
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_tickers_from_csv(path: Path) -> list[str]:
    # 文字コードの差異に強く（BOM あり/なし対応）
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 想定されるカラム名の候補（ユーザー指定: 'tickerｄ'）
    candidates = [
        "tickerｄ",
        "tickerｄ".replace("ｄ", "d"),
        "ticker",
        "Ticker",
        "TICKER",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise KeyError(
            f"{path.name} にティッカー列が見つかりません。候補: {candidates} / 実列: {list(df.columns)}"
        )

    # 例: '1925.T' → '1925'
    def normalize(x: str) -> str:
        s = str(x).strip()
        if not s:
            return ""
        return s.split(".", 1)[0]

    tickers = [normalize(v) for v in df[col].astype(str).tolist() if str(v).strip()]
    # 重複除去（順序保持）
    seen = set()
    uniq = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:A_MAX]


def write_to_xlsm(codes: list[str], xlsm_path: Path):
    path = Path(xlsm_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel ファイルが見つかりません: {path}")

    wb = load_workbook(
        filename=str(xlsm_path), keep_vba=True, read_only=False, data_only=False
    )
    try:
        ws = wb[SHEET_NAME]
        # A1〜A30にセット、余りは空に
        for i in range(A_MAX):
            ws.cell(row=i + 1, column=1, value=(codes[i] if i < len(codes) else None))
        wb.save(str(xlsm_path))
    finally:
        try:
            wb.close()
        except Exception:
            pass


def main():
    wl = pick_latest_watchlist()
    codes = read_tickers_from_csv(wl)
    write_to_xlsm(codes, SNAPSHOT_XLSM)
    print(
        f"✅ {wl.name} から {len(codes)} 件を書き込み → {SNAPSHOT_XLSM.name} の {SHEET_NAME}!A1:A{A_MAX}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
