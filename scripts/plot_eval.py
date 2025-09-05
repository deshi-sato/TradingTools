import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(
        description="picks評価CSVの可視化（エクイティ＆日次リターン）"
    )
    ap.add_argument(
        "--csv", required=True, help="reports\\picks_eval_open_close.csv のパス"
    )
    ap.add_argument("--out-dir", default="reports", help="PNGの出力先フォルダ")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "portfolio_ret_pct" not in df.columns:
        raise KeyError("CSVに必要な列（date, portfolio_ret_pct）がありません。")

    # 前処理
    df["date"] = pd.to_datetime(df["date"])
    ret = (df["portfolio_ret_pct"] / 100.0).astype(float)
    equity = (1.0 + ret.fillna(0)).cumprod()

    # 1) エクイティカーブ
    plt.figure()
    plt.plot(df["date"], equity)
    plt.title("Equity Curve (Open → Close)")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.tight_layout()
    eq_png = out_dir / "equity_curve.png"
    plt.savefig(eq_png, dpi=150)
    plt.close()

    # 2) 日次リターン推移
    plt.figure()
    plt.plot(df["date"], ret)
    plt.title("Daily Portfolio Returns (Open → Close)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    ret_png = out_dir / "daily_returns.png"
    plt.savefig(ret_png, dpi=150)
    plt.close()

    print("作成しました：")
    print(eq_png.resolve())
    print(ret_png.resolve())


if __name__ == "__main__":
    main()
