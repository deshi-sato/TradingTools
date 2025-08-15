import pandas as pd
from datetime import datetime, timedelta
import xlwings as xw
import os
import configparser
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask
import mplfinance as mpf

app = Flask(__name__)

# フォント設定（Noto Sans CJK JP を使用）
plt.rcParams["font.family"] = "Yu Gothic"

EXCEL_PATH = "C:/Users/Owner/Documents/desshi_signal_viewer/デイトレ株価データ.xlsm"
TEMP_PATH = "C:/Users/Owner/Documents/desshi_signal_viewer/temp_デイトレ株価データ.xlsm"

# === .ini 管理設定 ===
INI_PATH = "desshi_signal_viewer.ini"


def get_latest_row_index():
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    try:
        return int(config["読み込み状態"]["latest_row_index"])
    except:
        return 0


def save_latest_row_index(index):
    config = configparser.ConfigParser()
    if not os.path.exists(INI_PATH):
        config["読み込み状態"] = {"latest_row_index": str(index)}
    else:
        config.read(INI_PATH)
        config["読み込み状態"]["latest_row_index"] = str(index)
    with open(INI_PATH, "w") as f:
        config.write(f)

def get_japan_market_today():
    now = datetime.now()
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now < market_start:
        # 9:00より前 → 前日を「今日」とする
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # 9:00以降 → 通常の今日
        return now.strftime("%Y-%m-%d")


def save_chart_5min(ticker, df, global_data_dict):

    prev_open = prev_high = prev_low = prev_close = None

    if ticker not in global_data_dict:
        print("global_data_dictが不正または空です。チャート作成スキップ。")
        return None

    # 5分足に変換
    df_resampled = (
        df.resample("5min", on="time")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    # ✅ resample後チェック
    if (
        df_resampled.empty
        or df_resampled[["open", "high", "low", "close"]].dropna().empty
    ):
        print(f"⚠️ {ticker} の5分足データが不正または空です。チャート作成スキップ。")
        print(f"   df_resampled.shape: {df_resampled.shape}")
        print(f"   df_resampled.columns: {df_resampled.columns.tolist()}")
        return None

    df_resampled.index.name = "Date"
    df_resampled.reset_index(inplace=True)
    df_resampled.set_index("Date", inplace=True)

    # テクニカル指標
    df_resampled["5MA"] = df_resampled["close"].rolling(window=5).mean()
    df_resampled["25MA"] = df_resampled["close"].rolling(window=25).mean()
    df_resampled["VWAP"] = (
        df_resampled["close"] * df_resampled["volume"]
    ).cumsum() / df_resampled["volume"].cumsum()

    # カラム名変換
    df_plot = df_resampled.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    line_len = len(df_plot)

    # 最終チェック
    if (
        df_plot.empty
        or df_plot[["Open", "High", "Low", "Close"]].isnull().all().any()
        or df_plot[["Open", "High", "Low", "Close"]].dropna().shape[0] < 3
    ):
        print(f"⚠️ {ticker} の描画データが不正（空 or NaN or 3本未満）。スキップ。")
        print(f"   df_plot.shape: {df_plot.shape}")
        print(f"   df_plot.columns: {df_plot.columns.tolist()}")
        print(
            f"   df_plot[['Open', 'High', 'Low', 'Close']].dropna().shape: {df_plot[['Open', 'High', 'Low', 'Close']].dropna().shape}"
        )
        return None

    # インジケーターがNaNだけでないかチェック関数
    def is_valid_series(s, min_count=3):
        return s.dropna().shape[0] >= min_count

    # 安全なインジケーターだけを描画に追加
    add_plots = []

    # 当日の日付（df は当日分だけ）
    today = df["time"].dt.date.iloc[0]
    yesterday_str = str(today - timedelta(days=1))

    # global_data_dictから最新日付の前日を取得
    daily_dict = global_data_dict[ticker]
    if isinstance(daily_dict, dict) and daily_dict:
        # 日付を降順でソートして最新日付を取得
        sorted_dates = sorted(daily_dict.keys(), reverse=True)
        if sorted_dates:
            yesterday_str = sorted_dates[1]

    # グローバルから該当データ取得
    prev_df = global_data_dict.get(ticker, {}).get(yesterday_str)

    # 🔽 当日（df_plot）の範囲を取得
    today_high = df_plot["High"].max()
    today_low = df_plot["Low"].min()

    # 🔽 前日データを取得
    daily_dict = global_data_dict[ticker]
    if not isinstance(daily_dict, dict):
        print(f"⚠️ {ticker} に対応する値が dict ではありません: {type(daily_dict)}")
        return None
    elif yesterday_str not in daily_dict:
        print(f"⚠️ {ticker} は存在するが {yesterday_str} のデータがありません")
        return None
    else:
        prev_df = daily_dict[yesterday_str]

        # 前日データが存在する場合のみ前日四本値を取得
        if not prev_df.empty:
            try:
                prev_open = prev_df["open"].iloc[0]
                prev_high = prev_df["high"].max()
                prev_low = prev_df["low"].min()
                prev_close = prev_df["close"].iloc[-1]

                # 値がNoneまたはNaNでないことを確認
                if (
                    pd.isna(prev_open)
                    or pd.isna(prev_high)
                    or pd.isna(prev_low)
                    or pd.isna(prev_close)
                ):
                    prev_open = prev_high = prev_low = prev_close = None
            except Exception as e:
                prev_open = prev_high = prev_low = prev_close = None

            # 🔽 チャート範囲に含まれるOHLCのみライン追加
            if (
                prev_open is not None
                and isinstance(prev_open, (int, float))
                and not pd.isna(prev_open)
                and today_low <= prev_open <= today_high
            ):
                try:
                    add_plots.append(
                        mpf.make_addplot(
                            [float(prev_open)] * line_len,
                            panel=0,
                            color="gray",
                            linestyle="--",
                            width=0.8,
                        )
                    )
                except (ValueError, TypeError) as e:
                    pass

            if (
                prev_close is not None
                and isinstance(prev_close, (int, float))
                and not pd.isna(prev_close)
                and today_low <= prev_close <= today_high
            ):
                try:
                    add_plots.append(
                        mpf.make_addplot(
                            [float(prev_close)] * line_len,
                            panel=0,
                            color="black",
                            linestyle="--",
                            width=0.8,
                        )
                    )
                except (ValueError, TypeError) as e:
                    pass

            if (
                prev_high is not None
                and isinstance(prev_high, (int, float))
                and not pd.isna(prev_high)
                and today_low <= prev_high <= today_high
            ):
                try:
                    add_plots.append(
                        mpf.make_addplot(
                            [float(prev_high)] * line_len,
                            panel=0,
                            color="red",
                            linestyle=":",
                            width=0.8,
                        )
                    )
                except (ValueError, TypeError) as e:
                    pass

            if (
                prev_low is not None
                and isinstance(prev_low, (int, float))
                and not pd.isna(prev_low)
                and today_low <= prev_low <= today_high
            ):
                try:
                    add_plots.append(
                        mpf.make_addplot(
                            [float(prev_low)] * line_len,
                            panel=0,
                            color="blue",
                            linestyle=":",
                            width=0.8,
                        )
                    )
                except (ValueError, TypeError) as e:
                    pass

    if is_valid_series(df_plot["VWAP"]):
        try:
            # VWAPデータが数値型であることを確認
            vwap_data = df_plot["VWAP"].dropna()
            if not vwap_data.empty and vwap_data.dtype in ["float64", "int64"]:
                add_plots.append(
                    mpf.make_addplot(df_plot["VWAP"], color="orange", linestyle="-.")
                )
        except Exception as e:
            pass

    filename = f"{ticker}_{datetime.now().strftime('%Y%m%d')}.png"
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    abs_path = os.path.join(static_dir, filename)
    web_path = f"static/{filename}"

    try:
        s = mpf.make_mpf_style(
            # 基本はdefaultの設定値を使う。
            base_mpf_style="default",
            # font.family を matplotlibに設定されている値にする。
            rc={"font.family": plt.rcParams["font.family"][0]},
        )
        fig, axes = mpf.plot(
            df_plot,
            type="candle",
            mav=(5, 25),
            style=s,
            addplot=add_plots,
            ylabel="株価",
            ylabel_lower="出来高",
            volume=True,
            figsize=(16, 6),
            returnfig=True,  # ← fig, axes を取得する
        )

        # 🔽 前日四本値を注釈として下に表示
        if (
            prev_df is not None
            and not prev_df.empty
            and "prev_open" in locals()
            and prev_open is not None
            and prev_high is not None
            and prev_low is not None
            and prev_close is not None
            and isinstance(prev_open, (int, float))
            and isinstance(prev_high, (int, float))
            and isinstance(prev_low, (int, float))
            and isinstance(prev_close, (int, float))
            and not pd.isna(prev_open)
            and not pd.isna(prev_high)
            and not pd.isna(prev_low)
            and not pd.isna(prev_close)
        ):
            try:
                text_str = (
                    f"前日 始: {float(prev_open):.2f}  高: {float(prev_high):.2f}  "
                    f"安: {float(prev_low):.2f}  終: {float(prev_close):.2f}"
                )
                axes[0].text(
                    0.01,
                    -0.18,  # 左下の少し下
                    text_str,
                    transform=axes[0].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                )
            except (ValueError, TypeError) as e:
                pass

        fig.savefig(abs_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ファイルが実際に保存されたかチェック
        if os.path.exists(abs_path):
            return web_path
        else:
            print(f"❌ {ticker} チャートファイルが保存されませんでした: {abs_path}")
            return None

    except Exception as e:
        print(f"❌ {ticker} チャート描画失敗: {e}")
        return None


def evaluate_stock_long(day_frames):
    score = {
        "trend": 0,
        "volume": 0,
        "break": 0,
        "close_pos": 0,
        "volatility": 0,
        "vol_level": 0,
    }

    if len(day_frames) < 3:
        return score  # 評価に必要な最低日数に満たない

    # 日付の降順で最新5日を抽出
    sorted_days = sorted(day_frames.keys(), reverse=True)
    recent_days = sorted_days[:5]
    frames = [day_frames[day] for day in recent_days if day in day_frames]

    # 日別の終値・高値・安値・出来高を取得
    closes = [df["close"].iloc[-1] for df in frames if not df.empty]
    highs = [df["high"].max() for df in frames if not df.empty]
    lows = [df["low"].min() for df in frames if not df.empty]
    volumes = [df["volume"].sum() for df in frames if not df.empty]

    # ①トレンド
    if len(highs) >= 3 and len(lows) >= 3:
        if highs[2] < highs[1] < highs[0] and lows[2] < lows[1] < lows[0]:
            score["trend"] = 2
        elif highs[1] < highs[0] or lows[1] < lows[0]:
            score["trend"] = 1

    # ②出来高変化（直近 vs 前日）
    if len(volumes) >= 2 and volumes[1] > 0:
        ratio = (volumes[0] - volumes[1]) / volumes[1]
        if ratio >= 0.2:
            score["volume"] = 2
        elif ratio >= 0.05:
            score["volume"] = 1

    # ③ブレイク位置（終値 vs 前日高値）
    if len(highs) >= 2 and len(closes) >= 1:
        diff = (closes[0] - highs[1]) / highs[1]
        if diff >= 0.005:
            score["break"] = 2
        elif abs(diff) < 0.005:
            score["break"] = 1

    # ④引け位置（終値が当日高値に近い）
    today_high = highs[0]
    today_close = closes[0]
    if today_high:
        diff = (today_high - today_close) / today_high
        if diff <= 0.005:
            score["close_pos"] = 2
        elif diff <= 0.01:
            score["close_pos"] = 1

    # ⑤ボラティリティ（当日）
    today_low = lows[0]
    if today_close > 0 and (today_high - today_low) / today_close >= 0.03:
        score["volatility"] = 1

    # ⑥出来高水準（過去4日平均と比べて1.5倍以上）
    if len(volumes) >= 5:
        avg_volume = sum(volumes[1:5]) / 4
        if volumes[0] >= avg_volume * 1.5:
            score["vol_level"] = 1

    return score


def evaluate_stock_short(day_frames):
    score = {
        "trend": 0,
        "volume": 0,
        "break": 0,
        "close_pos": 0,
        "volatility": 0,
        "vol_level": 0,
    }

    if len(day_frames) < 3:
        return score

    sorted_days = sorted(day_frames.keys(), reverse=True)
    recent_days = sorted_days[:5]
    frames = [day_frames[day] for day in recent_days if not day_frames[day].empty]

    closes = [df["close"].iloc[-1] for df in frames]
    highs = [df["high"].max() for df in frames]
    lows = [df["low"].min() for df in frames]
    volumes = [df["volume"].sum() for df in frames]

    # ①トレンド：高値・安値が連続で切り下げ
    if len(highs) >= 3 and len(lows) >= 3:
        if highs[2] > highs[1] > highs[0] and lows[2] > lows[1] > lows[0]:
            score["trend"] = 2
        elif highs[1] > highs[0] or lows[1] > lows[0]:
            score["trend"] = 1

    # ②出来高：前々日急増 → 前日急減
    if len(volumes) >= 3 and volumes[1] < volumes[2] and volumes[1] < volumes[0]:
        score["volume"] = 2
    elif len(volumes) >= 2 and volumes[0] < volumes[1]:
        score["volume"] = 1

    # ③ブレイク位置：終値が前日安値を下回る
    if len(lows) >= 2 and len(closes) >= 1:
        diff = (closes[0] - lows[1]) / lows[1]
        if diff <= -0.005:
            score["break"] = 2
        elif diff < 0:
            score["break"] = 1

    # ④引け位置：終値が当日安値に近い
    today_low = lows[0]
    today_close = closes[0]
    if today_low:
        diff = (today_close - today_low) / today_low
        if diff <= 0.005:
            score["close_pos"] = 2
        elif diff <= 0.01:
            score["close_pos"] = 1

    # ⑤ボラティリティ
    today_high = highs[0]
    if today_close > 0 and (today_high - today_low) / today_close >= 0.03:
        score["volatility"] = 1

    # ⑥出来高水準
    if len(volumes) >= 5:
        avg_volume = sum(volumes[1:5]) / 4
        if volumes[0] >= avg_volume * 1.5:
            score["vol_level"] = 1

    return score


def create_score_table_long(data_dict):
    score_table = []

    for ticker, day_frames in data_dict.items():
        try:
            sorted_days = sorted(day_frames.keys())
            if len(sorted_days) < 2:
                continue
            # スコアは「前日」までで評価
            score_target_days = {day: day_frames[day] for day in sorted_days[:-1]}

            score = evaluate_stock_long(score_target_days)
            total = sum(score.values())

            # 前日のデータを表示用に使用
            prev_day = sorted_days[-2]
            prev_df = day_frames[prev_day]

            score_table.append(
                {
                    "ticker": ticker,
                    "終値": prev_df["close"].iloc[-1],
                    "直近高値": prev_df["high"].max(),
                    "直近安値": prev_df["low"].min(),
                    "トレンド": score["trend"],
                    "出来量変化": score["volume"],
                    "ブレイク": score["break"],
                    "引け位置": score["close_pos"],
                    "ボラ": score["volatility"],
                    "出来高水準": score["vol_level"],
                    "合計スコア": total,
                }
            )
        except Exception as e:
            print(f"{ticker}（ロング）処理中にエラー: {e}")

    df_score = pd.DataFrame(score_table)
    df_score = df_score.sort_values(by="合計スコア", ascending=False).reset_index(
        drop=True
    )
    return df_score


def create_score_table_short(data_dict):
    score_table = []

    for ticker, day_frames in data_dict.items():
        try:
            sorted_days = sorted(day_frames.keys())
            if len(sorted_days) < 2:
                continue
            score_target_days = {day: day_frames[day] for day in sorted_days[:-1]}

            score = evaluate_stock_short(score_target_days)
            total = sum(score.values())

            prev_day = sorted_days[-2]
            prev_df = day_frames[prev_day]

            score_table.append(
                {
                    "ticker": ticker,
                    "終値": prev_df["close"].iloc[-1],
                    "直近高値": prev_df["high"].max(),
                    "直近安値": prev_df["low"].min(),
                    "トレンド": score["trend"],
                    "出来量変化": score["volume"],
                    "ブレイク": score["break"],
                    "引け位置": score["close_pos"],
                    "ボラ": score["volatility"],
                    "出来高水準": score["vol_level"],
                    "合計スコア": total,
                }
            )
        except Exception as e:
            print(f"{ticker}（ショート）処理中にエラー: {e}")

    df_score = pd.DataFrame(score_table)
    df_score = df_score.sort_values(by="合計スコア", ascending=False).reset_index(
        drop=True
    )
    return df_score
import sqlite3
from typing import Dict, Optional, List

def _latest_trade_date(conn: sqlite3.Connection, ticker: str) -> Optional[str]:
    cur = conn.execute("""
        SELECT date(datetime) AS d
        FROM minute_data
        WHERE ticker=?
        ORDER BY datetime DESC
        LIMIT 1
    """, (ticker,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None

def _latest_completed_trade_date(conn: sqlite3.Connection, ticker: str, min_bars: int = 332) -> Optional[str]:
    cur = conn.execute(
        """
        SELECT d FROM (
            SELECT date(datetime) AS d, COUNT(*) AS c
            FROM minute_data
            WHERE ticker=?
            GROUP BY d
        )
        WHERE c >= ?
        ORDER BY d DESC
        LIMIT 1
        """,
        (ticker, min_bars),
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None

def _count_minutes_of_day(conn: sqlite3.Connection, ticker: str, trade_date: str) -> int:
    cur = conn.execute("""
        SELECT COUNT(*)
        FROM minute_data
        WHERE ticker=? AND date(datetime)=?
    """, (ticker, trade_date))
    return int(cur.fetchone()[0])

def _daily_ohlcv(conn: sqlite3.Connection, ticker: str, trade_date: str) -> Optional[Dict[str, float]]:
    # 当日のOHLCV（プレースホルダ分足：OHLCがNULLかつVolume=0 は無視）
    open_row = conn.execute("""
        SELECT open FROM minute_data
        WHERE ticker=? AND date(datetime)=? AND open IS NOT NULL
        ORDER BY datetime ASC LIMIT 1
    """, (ticker, trade_date)).fetchone()
    close_row = conn.execute("""
        SELECT close FROM minute_data
        WHERE ticker=? AND date(datetime)=? AND close IS NOT NULL
        ORDER BY datetime DESC LIMIT 1
    """, (ticker, trade_date)).fetchone()
    high_row = conn.execute("""
        SELECT MAX(high) FROM minute_data
        WHERE ticker=? AND date(datetime)=? AND high IS NOT NULL
    """, (ticker, trade_date)).fetchone()
    low_row = conn.execute("""
        SELECT MIN(low) FROM minute_data
        WHERE ticker=? AND date(datetime)=? AND low IS NOT NULL
    """, (ticker, trade_date)).fetchone()
    vol_row = conn.execute("""
        SELECT COALESCE(SUM(volume),0) FROM minute_data
        WHERE ticker=? AND date(datetime)=? AND volume IS NOT NULL
    """, (ticker, trade_date)).fetchone()

    if not any([open_row and open_row[0] is not None,
                close_row and close_row[0] is not None,
                high_row and high_row[0] is not None,
                low_row and low_row[0] is not None]):
        return None

    return {
        "open": float(open_row[0]) if open_row and open_row[0] is not None else None,
        "close": float(close_row[0]) if close_row and close_row[0] is not None else None,
        "high": float(high_row[0]) if high_row and high_row[0] is not None else None,
        "low":  float(low_row[0])  if low_row  and low_row[0]  is not None else None,
        "volume": int(vol_row[0]) if vol_row and vol_row[0] is not None else 0,
    }

def _prev_daily_refs(
    conn: sqlite3.Connection,
    ticker: str,
    base_date: str,
    n_days: int = 5,
    min_bars: int = 332,
) -> List[Dict[str, float]]:
    # base_date の前日から、分足本数が min_bars 以上の営業日だけを n_days 件取得
    cur = conn.execute(
        """
        SELECT d FROM (
            SELECT date(datetime) AS d, COUNT(*) AS c
            FROM minute_data
            WHERE ticker=? AND date(datetime) < ?
            GROUP BY d
        )
        WHERE c >= ?
        ORDER BY d DESC
        LIMIT ?
        """,
        (ticker, base_date, min_bars, n_days),
    )
    dates = [r[0] for r in cur.fetchall()]
    out: List[Dict[str, float]] = []
    for d in dates:
        v = _daily_ohlcv(conn, ticker, d)
        if v and all(v[k] is not None for k in ("open", "close", "high", "low")):
            out.append({"date": d, **v})
    return out

def _score_buy(today, prevs) -> int:
    # 買い（プラス）— 複数観点の多段階評価
    score = 0
    highs  = [p["high"]  for p in prevs[:5]]
    lows   = [p["low"]   for p in prevs[:5]]
    vols   = [p["volume"] for p in prevs[:5]]

    # 1) トレンド
    if len(highs) >= 3 and len(lows) >= 3:
        if highs[2] < highs[1] < highs[0] and lows[2] < lows[1] < lows[0]:
            score += 2
        elif highs[1] < highs[0] or lows[1] < lows[0]:
            score += 1

    # 2) 出来高（前日比：直近5〜10分急増の厳密判定は実装しない）
    if len(vols) >= 1 and vols[0] > 0:
        ratio = (today["volume"] - vols[0]) / max(vols[0], 1)
        if ratio >= 0.20:
            score += 2
        elif ratio >= 0.05:
            score += 1

    # 3) ブレイク位置（前日高値）
    if len(highs) >= 1 and today["close"] and highs[0]:
        diff = (today["close"] - highs[0]) / highs[0]
        if diff >= 0.005:
            score += 2
        elif abs(diff) < 0.005:
            score += 1

    # 4) 引け位置（当日高値に近い）
    if today["high"] and today["close"]:
        diff_close_high = (today["high"] - today["close"]) / today["high"]
        if diff_close_high <= 0.005:
            score += 2
        elif diff_close_high <= 0.01:
            score += 1

    # 5) ボラティリティ
    if today["close"] and (today["high"] - today["low"]) / today["close"] >= 0.03:
        score += 1

    # 6) 出来高水準（過去5日平均の1.5倍以上）
    if len(prevs) >= 5:
        avg5 = sum(p["volume"] for p in prevs[:5]) / 5
        if today["volume"] >= 1.5 * avg5:
            score += 1

    return int(score)

def _score_sell(today, prevs) -> int:
    # 売り（プラスで返却。最終的に負符号にする）
    score = 0
    highs  = [p["high"]  for p in prevs[:5]]
    lows   = [p["low"]   for p in prevs[:5]]
    vols   = [p["volume"] for p in prevs[:5]]

    # 1) トレンド：切り下げ
    if len(highs) >= 3 and len(lows) >= 3:
        if highs[2] > highs[1] > highs[0] and lows[2] > lows[1] > lows[0]:
            score += 2
        elif highs[1] > highs[0] or lows[1] > lows[0]:
            score += 1

    # 2) 出来高：急増→減少（近似）
    if len(vols) >= 3 and vols[1] < vols[2] and vols[1] < vols[0]:
        score += 2
    elif len(vols) >= 2 and vols[0] < vols[1]:
        score += 1

    # 3) ブレイク位置：前日安値割れ
    if len(lows) >= 1 and today["close"] and lows[0]:
        diff = (today["close"] - lows[0]) / lows[0]
        if diff <= -0.005:
            score += 2
        elif diff < 0:
            score += 1

    # 4) 引け位置：当日安値に近い
    if today["low"] and today["close"]:
        diff_close_low = (today["close"] - today["low"]) / today["low"]
        if diff_close_low <= 0.005:
            score += 2
        elif diff_close_low <= 0.01:
            score += 1

    # 5) ボラティリティ
    if today["close"] and (today["high"] - today["low"]) / today["close"] >= 0.03:
        score += 1

    # 6) 出来高水準
    if len(prevs) >= 5:
        avg5 = sum(p["volume"] for p in prevs[:5]) / 5
        if today["volume"] >= 1.5 * avg5:
            score += 1

    return int(score)

def compute_trend_score_for_snapshots(db_path: str) -> Dict[str, Optional[int]]:
    """
    Return {ticker: score or None} for all tickers in quote_latest.
    - Use the latest trade date that has at least 332 1-min bars.
    - Buy score and Sell score are computed separately.
      Final score = (buy >= sell) ? +buy : -sell
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out: Dict[str, Optional[int]] = {}

    tickers = [r[0] for r in conn.execute("SELECT ticker FROM quote_latest").fetchall()]

    for ticker in tickers:
        # 最新の「332本以上が揃っている」営業日を採用
        trade_date = _latest_completed_trade_date(conn, ticker, min_bars=332)
        if not trade_date:
            out[ticker] = None
            continue

        today = _daily_ohlcv(conn, ticker, trade_date)
        if not today or any(today[k] is None for k in ("open","close","high","low")):
            out[ticker] = None
            continue

        prevs = _prev_daily_refs(conn, ticker, trade_date, n_days=5, min_bars=332)
        if not prevs:
            out[ticker] = None
            continue

        buy = _score_buy(today, prevs)
        sell = _score_sell(today, prevs)
        out[ticker] = buy if buy >= sell else -sell

    conn.close()
    return out
