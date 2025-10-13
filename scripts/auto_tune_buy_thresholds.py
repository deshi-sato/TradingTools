# scripts/auto_tune_buy_thresholds.py
import argparse, json, subprocess, sys, time, os, csv
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



def _to_float(value, default=None):
    try:
        if value is None:
            return default
        value_str = str(value).strip()
        if not value_str:
            return default
        return float(value_str)
    except (TypeError, ValueError):
        return default


def _select_constrained_candidate(
    csv_path: str,
    dataset_id: str,
    round_index: int,
    base_uptick: float,
    base_spread: float,
    base_score: float,
    target_trades: int,
):
    csv_file = Path(csv_path) if csv_path else None
    if not csv_file or not csv_file.exists():
        return None, None

    best_row = None
    best_key = None
    with csv_file.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            trades = int(_to_float(row.get('trades'), 0) or 0)
            if trades < target_trades:
                continue

            uptick = _to_float(row.get('BUY_UPTICK_THR'), float('inf'))
            spread = _to_float(row.get('BUY_SPREAD_MAX'), float('inf'))
            score = _to_float(row.get('BUY_SCORE_THR'), float('inf'))

            if uptick is not None and uptick > base_uptick + 1e-9:
                continue
            if spread is not None and spread > base_spread + 1e-9:
                continue
            if score is not None and score > base_score + 1e-9:
                continue

            precision = _to_float(row.get('precision'), float('-inf'))
            ev = _to_float(row.get('ev'), float('-inf'))
            key = (precision, ev, trades)
            if best_row is None or key > best_key:
                best_row = row
                best_key = key

    if best_row is None:
        return None, None

    precision = _to_float(best_row.get('precision'), 0.0)
    ev = _to_float(best_row.get('ev'), 0.0)
    trades = int(_to_float(best_row.get('trades'), 0) or 0)
    mean_hit = _to_float(best_row.get('mean_hit_bp'), None)
    mean_miss = _to_float(best_row.get('mean_loss_bp'), None)

    params: dict[str, float] = {}
    for key in ('BUY_UPTICK_THR', 'BUY_SPREAD_MAX', 'BUY_SCORE_THR', 'COOLDOWN_SEC', 'VOL_SURGE_MIN'):
        val = best_row.get(key)
        num = _to_float(val, None)
        if num is not None:
            params[key] = num

    payload = {
        'dataset_id': dataset_id,
        'precision': precision,
        'trades': trades,
        'ev': ev,
        'mean_hit_bp': mean_hit,
        'mean_loss_bp': mean_miss,
        'params': params,
        'eligible': True,
    }

    out_path = Path('exports') / f"best_thresholds_filtered_{dataset_id}_round{round_index}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, str(out_path)


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
    current_uptick = float(uptick)
    current_spread = float(spread)
    current_score = float(score)
    current_ev_floor = float(args.EVfloor)
    best = None
    history = []

    for r in range(1, rounds + 1):
        uptick_r = round(current_uptick, 3)
        spread_r = round(current_spread)
        score_r = round(current_score, 3)

        print(
            f"\n==== Round {r}/{rounds}  target_trades>={target}  "
            f"[uptick>={uptick_r}, spread<={spread_r}, score>={score_r}] ===="
        )


        res = run_grid(
            dataset_id=ds,
            cv=args.CV,
            buy_uptick=uptick_r,
            spread_max=spread_r,
            score_thr=score_r,
            horizons=args.Horizons,
            min_trades=target,
            ev_floor=current_ev_floor,
            verbose=True,
        )

        best_json_path = res.get("json", "")
        best_json = read_best(best_json_path)
        if not best_json:
            print("[auto] best json not found; moving to next round.")
            continue

        constrained_payload, constrained_json_path = _select_constrained_candidate(
            res.get("csv", ""),
            dataset_id=ds,
            round_index=r,
            base_uptick=uptick_r,
            base_spread=spread_r,
            base_score=score_r,
            target_trades=target,
        )
        if constrained_payload:
            print("[auto] applied base-threshold constraints to grid results.")
            best_json = constrained_payload
            if constrained_json_path:
                best_json_path = constrained_json_path
        else:
            print("[auto] no candidate satisfied base-threshold constraints; using global best.")

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
                "json": best_json_path,
            }
        )

        print(
            f"[auto] round#{r} result -> trades={trades}  EV={ev}  prec={prec}  eligible={eligible}"
        )
        print(f"[auto] params={params}")

        if trades is not None and trades >= target and eligible:
            best = history[-1]
            print("\n[auto] target met; stopping auto-tune.")
            break

        if best is None or best.get("ev") is None or (ev is not None and ev > best.get("ev", float("-inf"))):
            best = history[-1]

        adjustments = []

        if trades is not None and trades < target:
            prev_spread = current_spread
            step = max(1, int((target - trades) / 50))
            current_spread = min(current_spread + step * 2, 40)
            if current_spread != prev_spread:
                adjustments.append(f"spread {prev_spread}->{current_spread}")

        if ev is not None and ev < current_ev_floor:
            prev_uptick = current_uptick
            prev_score = current_score
            prev_ev_floor = current_ev_floor
            current_uptick = min(current_uptick + 0.05, 0.95)
            current_score = min(current_score + 0.5, 12.0)
            current_ev_floor = max(current_ev_floor - 0.5, -20.0)
            adjustments.append(f"uptick {prev_uptick}->{current_uptick}")
            adjustments.append(f"score {prev_score}->{current_score}")
            adjustments.append(f"EVfloor {prev_ev_floor}->{current_ev_floor}")

        if not adjustments:
            print("[auto] no parameter adjustments this round.")
        else:
            print("[auto] adjustments: " + "; ".join(adjustments))

    else:
        if best is None and history:
            best = max(history, key=lambda r: (r.get("precision") or 0.0, r.get("ev") or float("-inf")))
    print("\n===== Auto Tune Summary =====")
    if best:
        print(f"best round={best['round']} trades={best['trades']} EV={best['ev']} precision={best['precision']}")
        print(f"best params={best['params']}")
        print(f"best json={best['json']}")
    else:
        print('No candidate reached the target criteria.')

    out = Path('exports') / f"auto_tune_summary_{ds}.json"
    out.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[auto] summary saved -> {out}')



if __name__ == "__main__":
    main()
