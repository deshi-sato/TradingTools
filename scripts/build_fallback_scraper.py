import time, csv, argparse, os, sys, json, re, subprocess, glob
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Selenium は --no-browser が無い時だけ使う ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://finance.matsui.co.jp"
URL  = BASE + "/ranking-day-trading-morning/index?page={}"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0.0.0 Safari/537.36")

ALNUM4_RE    = re.compile(r"\b[0-9A-Z]{4}\b")           # 英数字4桁（新規上場の英字コード対応）
HREF_CODE_RE = re.compile(r"/stock/([0-9A-Za-z]{4})/index")

# =============== utils ===============
def load_filter(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def pass_filter(row, filters):
    code = row["code"].upper()
    name = row["name"]
    price = row.get("price", 0)
    # exclude_codes は大文字比較
    if "exclude_codes" in filters and code in [c.upper() for c in filters["exclude_codes"]]:
        return False
    if "exclude_keywords" in filters:
        for kw in filters["exclude_keywords"]:
            if kw and kw in name:
                return False
    if "min_price" in filters and price < filters["min_price"]:
        return False
    if "max_price" in filters and price > filters["max_price"]:
        return False
    return True

def to_int(s: str) -> int:
    if not s: return 0
    s = s.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    m = re.search(r"[\d,]+", s)
    return int(m.group(0).replace(",", "")) if m else 0

def parse_yymmddhhmm(text: str):
    """
    例: '2025/09/22 15:30' -> '202509221530'
    フォーマットが取れない場合は None を返す
    """
    if not text:
        return None
    text = text.strip()
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2})", text)
    if not m:
        return None
    y, mo, d, h, mi = m.groups()
    return f"{y}{mo}{d}{h}{mi}"

def ensure_dir(path: str):
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def latest_fallback_csv(outdir: str) -> str | None:
    # 直近の fallback_daytrade_*.csv を探す
    pattern = os.path.join(outdir, "fallback_daytrade_*.csv")
    cand = sorted(glob.glob(pattern))
    return cand[-1] if cand else None

# =============== Selenium route (optional) ===============
def setup_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,1800")
    opts.add_argument("--lang=ja-JP")
    # ログ抑制 & bot検知軽減
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    service = Service(log_output=subprocess.DEVNULL)
    driver = webdriver.Chrome(options=opts, service=service)
    try:
        driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": UA})
    except Exception:
        pass
    return driver

def click_cookie_if_present(driver):
    try:
        for by, sel in [
            (By.XPATH, "//button[contains(.,'同意')]"),
            (By.XPATH, "//button[contains(.,'OK')]"),
            (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
            (By.CSS_SELECTOR, "div[class*='cookie'] button"),
        ]:
            btns = driver.find_elements(by, sel)
            if btns:
                btns[0].click(); time.sleep(0.2); break
    except Exception:
        pass

def parse_listing_rows_selenium(driver):
    """
    一覧テーブル（m-table[data-type='rankingDt']）から
    code(英数字4桁), href, price を収集。銘柄名は詳細ページで解決。
    """
    table = driver.find_element(By.CSS_SELECTOR, "table.m-table[data-type='rankingDt']")
    trs = table.find_elements(By.CSS_SELECTOR, "tbody tr")
    rows = []
    for tr in trs:
        tds = tr.find_elements(By.CSS_SELECTOR, "td")
        if len(tds) < 2:
            continue
        a = tds[1].find_element(By.TAG_NAME, "a")
        span = tds[1].find_element(By.TAG_NAME, "span")

        href = a.get_attribute("href") or ""
        code = ""
        m = HREF_CODE_RE.search(href)
        if m:
            code = m.group(1).upper()
        if not code:
            m2 = ALNUM4_RE.search((span.text or "").upper())
            if m2:
                code = m2.group(0)

        price = to_int(tds[2].text) if len(tds) >= 3 else 0
        if code and href:
            rows.append({"code": code, "href": href, "price": price})
    return rows

def extract_timestamp_from_dom_selenium(driver) -> str | None:
    """
    右上の更新時刻: div.m-table-utils-right > div.m-table-utils-refresh > p
    を拾って YYYYMMDDHHMM に整形
    """
    try:
        el = driver.find_element(By.CSS_SELECTOR, "div.m-table-utils-right div.m-table-utils-refresh p")
        return parse_yymmddhhmm(el.text)
    except Exception:
        return None

# =============== Requests route (no-browser) ===============
def fetch_listing_html(page: int) -> str:
    url = URL.format(page)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=10)
    r.raise_for_status()
    return r.text

def parse_listing_rows_from_html(html: str):
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.m-table[data-type='rankingDt']")
    rows = []
    if not table:
        return rows
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        a = tds[1].find("a")
        span = tds[1].find("span")
        if not a or not span:
            continue
        href = a.get("href") or ""
        code = ""
        m = HREF_CODE_RE.search(href)
        if m:
            code = m.group(1).upper()
        if not code:
            m2 = ALNUM4_RE.search((span.get_text(" ") or "").upper())
            if m2:
                code = m2.group(0)
        price = to_int(tds[2].get_text(" ")) if len(tds) >= 3 else 0
        if code and href:
            rows.append({"code": code, "href": href, "price": price})
    return rows

def extract_timestamp_from_html(html: str) -> str | None:
    """
    HTMLから更新時刻の<p>を拾って YYYYMMDDHHMM に整形
    """
    soup = BeautifulSoup(html, "lxml")
    p = soup.select_one("div.m-table-utils-right div.m-table-utils-refresh p")
    return parse_yymmddhhmm(p.get_text(" ").strip() if p else "")

# =============== name resolving ===============
def fetch_with_retry(url: str, timeout=10, max_retry=3):
    for i in range(max_retry):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
            if r.status_code == 200 and r.text:
                return r
        except Exception:
            pass
        time.sleep(0.4 + 0.3 * i)  # 軽いバックオフ
    raise RuntimeError(f"GET failed: {url}")

def extract_name_from_title(title: str) -> str:
    if not title: return ""
    title = title.strip()
    # "(XXXX)" の XXXX を英数字4桁で認識
    m = re.search(r"^(.+?)\s*[\(（]\s*[0-9A-Za-z]{4}\s*[\)）]", title)
    if m:
        return m.group(1).strip()
    m = re.search(r"^(.+?)\s*(?:の株価|\|)", title)
    if m:
        return m.group(1).strip()
    return title

def resolve_name_via_detail(href: str) -> str:
    url = href if href.startswith("http") else (BASE + href)
    r = fetch_with_retry(url)
    soup = BeautifulSoup(r.text, "html.parser")
    meta = soup.find("meta", attrs={"property": "og:title"})
    title = (meta.get("content") if meta and meta.get("content") else
             soup.title.string if soup.title else "")
    return extract_name_from_title(title)

def worker_resolve(row):
    code, href = row["code"], row["href"]
    name = resolve_name_via_detail(href)
    return {"code": code, "name": name, "price": row.get("price", 0)}

# =============== main ===============
def main():
    parser = argparse.ArgumentParser()
    # ★ fixed 機能は撤去。出力は「日付入りファイル名」を data/ に残す
    parser.add_argument("--outdir",   default="data", help="出力ディレクトリ")
    parser.add_argument("--filter",   default="config/fallback_filter.json")
    parser.add_argument("--encoding", default="utf-8-sig",  # Excel対策: UTF-8(BOM)
                        help="CSV encoding (utf-8-sig 推奨 / cp932 も可)")
    parser.add_argument("--workers",  type=int, default=10, help="parallel workers for name resolving")
    parser.add_argument("--no-browser", action="store_true", help="requestsのみで取得（Selenium/Chromeを起動しない）")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    fetched, fallback_used = 0, False
    driver = None
    ts_from_site = None
    out_csv_path = None

    try:
        filters = load_filter(args.filter)

        # --- 一覧取得（no-browser なら requests、そうでなければ headless Selenium） ---
        listing = []
        pages = (1, 2)
        if args.no_browser:
            html_all = []
            for p in pages:
                html = fetch_listing_html(p)
                html_all.append(html)
                listing.extend(parse_listing_rows_from_html(html))
            # 1ページ目のHTMLから更新時刻を拾う（無ければ None）
            ts_from_site = extract_timestamp_from_html(html_all[0]) if html_all else None
        else:
            driver = setup_driver(headless=True)  # 画面は出ない
            for p in pages:
                driver.get(URL.format(p))
                WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
                click_cookie_if_present(driver)
                driver.execute_script("window.scrollTo(0, 600);")
                time.sleep(0.3)
                listing.extend(parse_listing_rows_selenium(driver))
            ts_from_site = extract_timestamp_from_dom_selenium(driver)

        # --- 銘柄名解決を並列化 ---
        results = []
        if listing:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futures = [ex.submit(worker_resolve, row) for row in listing]
                for fu in as_completed(futures):
                    try:
                        resolved = fu.result()
                        if pass_filter(resolved, filters):
                            results.append([resolved["code"].upper(), resolved["name"], "matsui_morning"])
                    except Exception:
                        continue

        fetched = len(results)
        if fetched == 0:
            raise RuntimeError("No rows fetched")

        # ---- 出力ファイル名を決定（サイトの更新時刻が取れなければ現在時刻で代用） ----
        ts = ts_from_site or time.strftime("%Y%m%d%H%M")  # e.g. 202509221530
        ensure_dir(args.outdir)
        out_csv_path = os.path.join(args.outdir, f"fallback_daytrade_{ts}.csv")

        # CSV書き込み（履歴は消さない）
        with open(out_csv_path, "w", newline="", encoding=args.encoding) as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(["code", "name", "reason"])
            w.writerows(results)

    except Exception as e:
        print(f"[WARN] Using latest existing fallback (reason=fetch_error) {e}", file=sys.stderr)
        fallback_used = True
        ensure_dir(args.outdir)
        last = latest_fallback_csv(args.outdir)
        if last:
            out_csv_path = last
        else:
            # 履歴ゼロで取得も失敗した場合は空運用を避けるためエラー終了
            print("[ERROR] No fallback_daytrade_*.csv found to fall back to.", file=sys.stderr)
            sys.exit(2)
    finally:
        if driver:
            driver.quit()

    elapsed_ms = int((time.time() - t0) * 1000)
    # 採用ファイル（新規作成or既存最新）のパスを明示
    print(f"summary fetched={fetched} fallback_used={fallback_used} path={out_csv_path} elapsed_ms={elapsed_ms}")

if __name__ == "__main__":
    main()
