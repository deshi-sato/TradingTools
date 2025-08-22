import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

DB_PATH = "rss_daily.db"  # 日足データベース
TABLE = "stock_prices"  # テーブル名（score_tuner.py と同じ）
DATE_COL = "date"
CODE_COL = "ticker"
PRICE_COL = "close"

# === ウィンドウ設定 ===
TRAIN_DAYS = 180
TEST_DAYS = 30


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT {DATE_COL}, {CODE_COL}, {PRICE_COL} FROM {TABLE}", conn)
    conn.close()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def walkforward(df):
    results = []
    start_date = df[DATE_COL].min()
    end_date = df[DATE_COL].max()

    current_train_start = start_date
    while True:
        train_start = current_train_start
        train_end = train_start + timedelta(days=TRAIN_DAYS - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=TEST_DAYS - 1)

        if test_end > end_date:
            break

        # 学習データ・検証データ
        df_train = df[(df[DATE_COL] >= train_start) & (df[DATE_COL] <= train_end)]
        df_test = df[(df[DATE_COL] >= test_start) & (df[DATE_COL] <= test_end)]

        # 簡易スコア例：直近180日リターンが大きい銘柄ほど強いと仮定
        train_returns = df_train.groupby(CODE_COL)[PRICE_COL].apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1
        )
        df_test = (
            df_test.groupby(CODE_COL)
            .apply(lambda g: g[PRICE_COL].iloc[-1] / g[PRICE_COL].iloc[0] - 1)
            .reset_index()
        )
        df_test.columns = [CODE_COL, "test_ret"]

        # trainランキング上位と test成績の対応
        top_codes = train_returns.sort_values(ascending=False).head(10).index
        avg_test_ret = df_test[df_test[CODE_COL].isin(top_codes)]["test_ret"].mean()

        results.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "avg_test_ret": avg_test_ret,
            }
        )

        # スライド（検証期間分進める）
        current_train_start = current_train_start + timedelta(days=TEST_DAYS)

    return pd.DataFrame(results)


def plot_results(results):
    results["cum_ret"] = (1 + results["avg_test_ret"]).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(results["test_end"], results["cum_ret"], marker="o")
    plt.title("Walkforward cumulative return (Top10 strategy)")
    plt.xlabel("Test period end")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    plt.savefig("walkforward_cum.png")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    results = walkforward(df)
    results.to_csv("walkforward_results.csv", index=False)
    plot_results(results)
    print("✅ walkforward_results.csv / walkforward_cum.png を出力しました")
