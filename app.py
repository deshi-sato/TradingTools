#
# Flask サーバー
# 2025.07.31
#
from flask import Flask, render_template, request, jsonify
from score_table import (
    load_data,
    create_score_table_long,
    create_score_table_short,
    load_summary_data,
    export_sheets,
    save_chart_5min,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import subprocess
from datetime import datetime, timedelta
import subprocess


EXCEL_EXE = r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "デイトレ株価データ.xlsm")
EXCEL_PATH_L = os.path.join(BASE_DIR, "買い銘柄寄り後情報.xlsm")
EXCEL_PATH_R = os.path.join(BASE_DIR, "売り銘柄寄り後情報.xlsm")
CHART_DIR = "static/charts"

GLOBAL_DATA_DICT = {}
CODE_TO_NAME = {}
step_mode = 0


def is_marketspeed_running_cmd():
    result = subprocess.run(["tasklist"], capture_output=True, text=True)

    #    print("🧪 tasklist 出力の一部を表示:")
    #    print(result.stdout[:1000])  # 上位1000文字だけ表示（長すぎる場合に備えて）

    if "marketspeed2.exe" in result.stdout.lower():
        print("✅ MarketSpeed2.exe が検出されました")
        return True
    else:
        print("❌ MarketSpeed2.exe は実行されていません")
        return False


def is_excel_open_recently(file_path, threshold_minutes=2):
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        now = datetime.now()
        return (now - mtime) < timedelta(minutes=threshold_minutes)
    except Exception as e:
        print(f"⚠️ タイムスタンプ取得エラー: {file_path} -> {e}")
        return False


def filter_top(df, min_count=5):
    grouped = df.groupby("合計スコア").size().sort_index(ascending=False)
    total = 0
    threshold = 0
    for score, count in grouped.items():
        total += count
        threshold = score
        if total >= min_count:
            break
    return df[df["合計スコア"] >= threshold]


@app.before_request
def initialize_once():
    import time

    global long_top, short_top, GLOBAL_DATA_DICT, CODE_TO_NAME, step_mode
    if request.endpoint != "index":
        return
    if not is_marketspeed_running_cmd():
        print("⚠️ MARKET SPEED2 が起動していません")
        return "<h2>MARKET SPEED2 を起動してください</h2>"
    if not hasattr(app, "ini_initialized"):
        app.ini_initialized = True

        GLOBAL_DATA_DICT, CODE_TO_NAME = load_data(EXCEL_PATH)

        long_df = create_score_table_long(GLOBAL_DATA_DICT)
        short_df = create_score_table_short(GLOBAL_DATA_DICT)

        long_top = filter_top(long_df, min_count=5)
        short_top = filter_top(short_df, min_count=5)

        export_sheets(EXCEL_PATH, long_top, short_top, CODE_TO_NAME)

        try:
            subprocess.Popen([EXCEL_EXE, EXCEL_PATH_L])
            subprocess.Popen([EXCEL_EXE, EXCEL_PATH_R])
        except Exception as e:
            print("Excel 起動エラー:", e)
    time.sleep(10)
    step_mode = 1


@app.route("/charts")
def charts():
    global GLOBAL_DATA_DICT, CODE_TO_NAME, step_mode
    try:
        if not step_mode:
            return jsonify([])
        if not is_excel_open_recently(EXCEL_PATH_L) or not is_excel_open_recently(
            EXCEL_PATH_R
        ):
            print("⏳ Excelファイルは更新直後のため /charts をスキップします")
            return jsonify([])

        chart_data = []
        combined_l, name_l = load_summary_data(EXCEL_PATH_L)
        combined_r, name_r = load_summary_data(EXCEL_PATH_R)

        # ✅ 通信未確立などで空の場合はスキップ
        if not combined_l and not combined_r:
            print("⚠️ load_summary_data によりデータ取得できず /charts スキップ")
            return jsonify([])

        combined = {**combined_l, **combined_r}
        name_map = {**name_l, **name_r}

        for ticker, daily_data in combined.items():
            try:
                # 最新日付のデータを取得
                latest_date = list(daily_data.keys())[0]
                df = daily_data[latest_date]
                chart_path = save_chart_5min(ticker, df, GLOBAL_DATA_DICT)
                if chart_path:
                    chart_data.append(
                        {
                            "ticker": ticker,
                            "ticker_name": name_map.get(ticker, ticker),
                            "img_url": f"/{chart_path}",
                        }
                    )
            except Exception as e:
                print(f"⚠️ {ticker} のチャート作成でエラー: {e}")
                continue

        return jsonify(chart_data)

    except Exception as e:
        print(f"❌ /charts ルート処理中に例外発生: {e}")
        return jsonify([])


@app.route("/")
def index():
    global GLOBAL_DATA_DICT, CODE_TO_NAME, step_mode
    try:
        if not step_mode or not GLOBAL_DATA_DICT:
            return "<h2>初期化中</h2>"
        if not is_excel_open_recently(EXCEL_PATH_L) or not is_excel_open_recently(
            EXCEL_PATH_R
        ):
            print("⏳ Excelファイルは更新直後のため index をスキップします")
            return "<h2>読み込み中</h2>"

        charts_5min = []
        combined_l, name_l = load_summary_data(EXCEL_PATH_L)
        combined_r, name_r = load_summary_data(EXCEL_PATH_R)

        # ✅ 通信未確立などで空の場合はスキップ
        if not combined_l and not combined_r:
            print("⚠️ load_summary_data によりデータ取得できず index スキップ")
            return "<h2>データ取得待ち（通信未確立）</h2>"

        combined = {**combined_l, **combined_r}
        name_map = {**name_l, **name_r}

        for ticker, daily_data in combined.items():
            try:
                # 最新日付のデータを取得
                latest_date = list(daily_data.keys())[0]
                df = daily_data[latest_date]
                chart_path = save_chart_5min(ticker, df, GLOBAL_DATA_DICT)
                if chart_path:
                    charts_5min.append(
                        (ticker, name_map.get(ticker, ticker), chart_path)
                    )
            except Exception as e:
                print(f"⚠️ {ticker} のチャート作成でエラー: {e}")
                continue

        return render_template("index.html", charts_5min=charts_5min)

    except Exception as e:
        print(f"❌ index() ルート処理中に例外発生: {e}")
        return "<h2>チャート描画中にエラーが発生しました</h2>"


if __name__ == "__main__":
    app.run(debug=True)
