\"\"\"plot_eval_buy_sell.py : Auto-generated placeholder

- file: scripts/plot_eval_buy_sell.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def cumprod_from_pct(series_pct: pd.Series) -> pd.Series:
    """NaNは0として無視し、(1+r)の累積を返す（始値=1）。"""
    r = (series_pct.fillna(0.0) / 100.0).astype(float)
    return (1.0 + r).cumprod()


def line_plot(x, y, title: str, y_label: str, out_png: Path):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="picks評価CSVの可視化（ポート/BUY/SELL 個別）"
    )
    ap.add_argument(
        "--csv", required=True, help="reports\\picks_eval_open_close.csv のパス"
    )
    ap.add_argument("--out-dir", default="reports", help="PNG/CSVの出力先フォルダ")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(csv_path)
    need = {"date", "portfolio_ret_pct", "buy_ret_pct", "sell_ret_pct"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"CSVに必要な列がありません: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    # Daily returns
    port_ret = df["portfolio_ret_pct"]
    buy_ret = df["buy_ret_pct"]
    sell_ret = df["sell_ret_pct"]

    # Equity (normalized start = 1)
    port_eq = cumprod_from_pct(port_ret)
    buy_eq = cumprod_from_pct(buy_ret)
    sell_eq = cumprod_from_pct(sell_ret)

    # --- PNGs ---
    line_plot(
        df["date"],
        port_eq,
        "Portfolio Equity Curve (Open → Close)",
        "Equity (normalized)",
        out_dir / "equity_portfolio.png",
    )
    line_plot(
        df["date"],
        buy_eq,
        "BUY-only Equity Curve (Open → Close)",
        "Equity (normalized)",
        out_dir / "equity_buy.png",
    )
    line_plot(
        df["date"],
        sell_eq,
        "SELL-only Equity Curve (Open → Close)",
        "Equity (normalized)",
        out_dir / "equity_sell.png",
    )

    line_plot(
        df["date"],
        port_ret / 100.0,
        "Daily Portfolio Returns",
        "Return",
        out_dir / "daily_returns_portfolio.png",
    )
    line_plot(
        df["date"],
        buy_ret / 100.0,
        "Daily BUY Returns",
        "Return",
        out_dir / "daily_returns_buy.png",
    )
    line_plot(
        df["date"],
        sell_ret / 100.0,
        "Daily SELL Returns",
        "Return",
        out_dir / "daily_returns_sell.png",
    )

    # --- Equity CSV（比較しやすいようにまとめて保存） ---
    eq_df = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "equity_portfolio": port_eq,
            "equity_buy": buy_eq,
            "equity_sell": sell_eq,
        }
    )
    eq_csv = out_dir / "equity_curves_buy_sell.csv"
    eq_df.to_csv(eq_csv, index=False, encoding="utf-8-sig")

    print("作成しました：")
    print((out_dir / "equity_portfolio.png").resolve())
    print((out_dir / "equity_buy.png").resolve())
    print((out_dir / "equity_sell.png").resolve())
    print((out_dir / "daily_returns_portfolio.png").resolve())
    print((out_dir / "daily_returns_buy.png").resolve())
    print((out_dir / "daily_returns_sell.png").resolve())
    print(eq_csv.resolve())


if __name__ == "__main__":
    main()
