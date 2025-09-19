# scripts/closeout_10am.py
from __future__ import annotations
import os, sys, json, glob, zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))


def _today_tag() -> str:
    return datetime.now(tz=JST).strftime("%Y%m%d")


def _read_jsonl(path: Path):
    rows = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return rows


def collect_orders() -> list[dict]:
    files = sorted(Path("logs").glob(f"orders-{_today_tag()}.jsonl"))
    rows: list[dict] = []
    for p in files:
        rows.extend(_read_jsonl(p))
    return rows


def summarize(rows: list[dict]) -> dict:
    touched = sorted({r.get("symbol") for r in rows if r.get("symbol")})
    ifdoco = sum(1 for r in rows if r.get("action") == "IFDOCO")
    rejects = sum(1 for r in rows if r.get("phase") == "reject")
    errors = sum(1 for r in rows if "ERROR" in json.dumps(r))
    buy = sum(1 for r in rows if r.get("side") == "BUY")
    sell = sum(1 for r in rows if r.get("side") == "SELL")
    return {
        "date": _today_tag(),
        "lines": len(rows),
        "symbols": touched,
        "symbols_count": len(touched),
        "ifdoco": ifdoco,
        "reject": rejects,
        "errors": errors,
        "buy": buy,
        "sell": sell,
    }


def try_live_close():
    """LIVE時だけ、ユーザー側ラッパーがあれば呼ぶ."""
    try:
        from exec.kabu_exec import close_all_positions, cancel_all_orders  # type: ignore
    except Exception:
        print(
            "[closeout] LIVE close skipped: exec.kabu_exec.{close_all_positions,cancel_all_orders} not found"
        )
        return {"close_called": False}
    try:
        cancel_all_orders()
    except Exception as e:
        print(f"[closeout] cancel_all_orders error: {e}")
    try:
        close_all_positions()
    except Exception as e:
        print(f"[closeout] close_all_positions error: {e}")
    return {"close_called": True}


def zip_logs():
    Path("archive").mkdir(exist_ok=True)
    tag = _today_tag()
    out = Path(f"archive/{tag}.zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in Path("logs").glob(f"*{tag}*"):
            z.write(p, p.as_posix())
        for p in Path("data").glob(f"*{tag}*"):
            z.write(p, p.as_posix())
    return str(out)


def main():
    mode = os.environ.get("MODE", "PAPER").upper()
    rows = collect_orders()
    summ = summarize(rows)
    summ["mode"] = mode

    if mode == "LIVE":
        summ.update(try_live_close())
    else:
        summ["close_called"] = False

    # サマリ保存
    Path("logs").mkdir(exist_ok=True)
    out = Path(f"logs/close_summary-{_today_tag()}.json")
    out.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[closeout] wrote summary -> {out}")

    # アーカイブ
    arc = zip_logs()
    print(f"[closeout] archived logs -> {arc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[closeout] fatal: {e}", file=sys.stderr)
        sys.exit(1)
