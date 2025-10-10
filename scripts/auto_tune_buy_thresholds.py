# scripts/auto_tune_buy_thresholds.py
import argparse, json, subprocess, sys, time, os
from pathlib import Path

"""
目的:
  - BUYグリッドサーチを自動で繰り返し、
    「MinTrades >= 目標件数」を満たすまで閾値レンジを段階的に緩める。
  - 各ラウンドで scripts.grid_search_thresholds を実行し、出力CSV/JSONを読み取って判定。
  - 最終的に best_thresholds_*.json を採用し、コンソールに要約を表示。

使い方:
  py -m scripts.auto_tune_buy_thresholds -DatasetId REF20251010_OFF -TargetTrades 10

前提:
  - scripts.grid_search_thresholds が動作すること
  - exports/grid_report_*.csv, exports/best_thresholds_*.json を書き出すこと
"""


def run_grid(
    dataset_id: str,
    cv: int,
    buy_uptick: float,
    spread_max: int,
    score_thr: float,
    horizons: str = "60,120",
    min_trades: int = 0,
    ev_floor: float = 0.0,
    verbose: bool = True,
) -> dict:
    # 1回分のグリッドサーチ実行（ここでは単点でなく、±αのレンジも同時に走らせる）
    #   -> 既存の scripts.grid_search_thresholds は内部で離散レンジを持っている想定
    #       ここは「起点値」を渡しつつ、内部の既定レンジに任せる前提でシンプルに。
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.grid_search_thresholds",
        "-DatasetId",
        dataset_id,
        "-Horizons",
        horizons,
        "-MinTrades",
        str(min_trades),
        "-EVFloor",
        str(ev_floor),
    ]
    # 進捗を見やすく
    if verbose:
        print(
            f"\n[auto] run grid: Dataset={dataset_id}  base(BUY_UPTICK_THR={buy_uptick}, BUY_SPREAD_MAX={spread_max}, BUY_SCORE_THR={score_thr})"
        )
    start = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if verbose:
        print(p.stdout)
        if p.stderr.strip():
            print("[stderr]", p.stderr)

    # 出力ファイル探索
    exp = Path("exports")
    # 最新の grid_report_*.csv / best_thresholds_*.json を拾う
    latest_csv = max(
        exp.glob("grid_report_*.csv"), key=lambda p: p.stat().st_mtime, default=None
    )
    latest_json = max(
        exp.glob("best_thresholds_*.json"),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )

    result = {
        "elapsed": elapsed,
        "csv": str(latest_csv) if latest_csv else "",
        "json": str(latest_json) if latest_json else "",
        "stdout": p.stdout,
        "stderr": p.stderr,
        "returncode": p.returncode,
    }
    return result


def read_best(json_path: str) -> dict:
    if not json_path or not Path(json_path).exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-DatasetId", required=True)
    ap.add_argument("-TargetTrades", type=int, default=10, help="目標トレード件数")
    ap.add_argument("-MaxRounds", type=int, default=6, help="緩める段階の最大回数")
    ap.add_argument("-Horizons", default="60,120")
    ap.add_argument("-BaseUptick", type=float, default=0.60)
    ap.add_argument("-BaseSpread", type=int, default=10)
    ap.add_argument("-BaseScore", type=float, default=6.0)
    ap.add_argument("-EVfloor", type=float, default=0.0)
    ap.add_argument("-CV", type=int, default=0)  # 0=NoCV, 5=CV5など
    args = ap.parse_args()

    ds = args.DatasetId
    target = args.TargetTrades
    rounds = args.MaxRounds

    # 起点（少し厳しめ→緩める方向）
    uptick = args.BaseUptick  # 例: 0.60
    spread = args.BaseSpread  # 例: 10
    score = args.BaseScore  # 例: 6.0

    # 段階的に緩めるロジック
    #   Roundごとに:
    #     - BUY_UPTICK_THR を -0.05
    #     - BUY_SPREAD_MAX を +2
    #     - BUY_SCORE_THR を -0.5
    #   の3軸を少しずつ解放（※下限・上限に注意）
    best = None
    history = []

    for r in range(1, rounds + 1):
        # ガード
        uptick_r = max(0.50, round(uptick - 0.05 * (r - 1), 3))  # 0.50未満はやりすぎ
        spread_r = min(20, spread + 2 * (r - 1))  # 20超は無効域想定
        score_r = max(4.0, round(score - 0.5 * (r - 1), 3))  # 4.0未満はノイズ拾いを警戒

        print(
            f"\n==== Round {r}/{rounds}  target_trades>={target}  "
            f"[uptick≥{uptick_r}, spread≤{spread_r}, score≥{score_r}] ===="
        )

        res = run_grid(
            dataset_id=ds,
            cv=args.CV,
            buy_uptick=uptick_r,
            spread_max=spread_r,
            score_thr=score_r,
            horizons=args.Horizons,
            min_trades=target,
            ev_floor=args.EVfloor,
            verbose=True,
        )

        best_json = read_best(res.get("json", ""))
        if not best_json:
            print("[auto] best json not found. 次ラウンドへ。")
            continue

        params = best_json.get("params", {})
        trades = best_json.get("trades", 0)
        ev = best_json.get("ev", None)
        prec = best_json.get("precision", None)
        eligible = best_json.get("eligible", False)

        history.append(
            {
                "round": r,
                "trades": trades,
                "ev": ev,
                "precision": prec,
                "params": params,
                "eligible": eligible,
                "json": res.get("json"),
            }
        )

        print(
            f"[auto] round#{r} result -> trades={trades}  EV={ev}  prec={prec}  eligible={eligible}"
        )
        print(f"[auto] params={params}")

        # 目標達成チェック
        if trades is not None and trades >= target and eligible:
            best = history[-1]
            print("\n[auto] ✅ 目標件数を満たす候補を確保しました。")
            break

        # 未達だがEVが良い案をメモ（最後の保険）
        if best is None:
            best = history[-1]

    # 最終採用の表示
    print("\n===== Auto Tune Summary =====")
    if best:
        print(
            f"採用 round={best['round']}  trades={best['trades']}  EV={best['ev']}  precision={best['precision']}"
        )
        print(f"採用 params={best['params']}")
        print(f"採用 json={best['json']}")
    else:
        print("候補を取得できませんでした。")

    # 履歴ログも保存
    out = Path("exports") / f"auto_tune_summary_{ds}.json"
    out.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[auto] 履歴を保存 -> {out}")


if __name__ == "__main__":
    main()
