#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
naut_view.py

raw_push テーブルに蓄積された板配信データを可視化するビューア。
リフィード DB と銘柄コードを引数に受け取り、TradingVolumeTime を基準に 1/3/5 分足へ集計した
ローソク足と出来高バーを描画する。カーソル位置の板サマリはチャート右側へ注記する。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from dateutil import parser as dateutil_parser  # type: ignore[import-untyped]


def _resolve_jst() -> tzinfo:
    try:
        from zoneinfo import ZoneInfo  # type: ignore[import]
    except ImportError:  # pragma: no cover - Python <3.9 fallback
        return timezone(timedelta(hours=9))
    try:
        return ZoneInfo("Asia/Tokyo")
    except Exception:
        return timezone(timedelta(hours=9))


JST = _resolve_jst()


@dataclass
class Snapshot:
    symbol: str
    t_recv: float
    recv_dt: datetime
    event_dt: Optional[datetime] = None
    symbol_name: Optional[str] = None
    interval_minutes: Optional[int] = None
    bar_start: Optional[datetime] = None
    bar_end: Optional[datetime] = None
    volume_time: Optional[datetime] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[float] = None
    cum_volume: Optional[float] = None
    buy_levels: Sequence[Tuple[Optional[float], Optional[float]]] = ()
    sell_levels: Sequence[Tuple[Optional[float], Optional[float]]] = ()
    over_sell: Optional[float] = None
    under_buy: Optional[float] = None
    total_bid: Optional[float] = None
    total_ask: Optional[float] = None
    seq: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def primary_dt(self) -> datetime:
        base = self.bar_end or self.volume_time or self.event_dt or self.recv_dt
        if base.tzinfo is None:
            return base.replace(tzinfo=JST)
        return base.astimezone(JST)

    def primary_dt_naive(self) -> datetime:
        return self.primary_dt().replace(tzinfo=None)

    def recv_dt_naive(self) -> datetime:
        dt = self.recv_dt.astimezone(JST) if self.recv_dt.tzinfo else self.recv_dt
        return dt.replace(tzinfo=None)

    def has_ohlc(self) -> bool:
        return (
            self.open_price is not None
            and self.high_price is not None
            and self.low_price is not None
            and self.close_price is not None
        )

    def bid_qty(self) -> Optional[float]:
        if self.buy_levels:
            return self.buy_levels[0][1]
        return None

    def ask_qty(self) -> Optional[float]:
        if self.sell_levels:
            return self.sell_levels[0][1]
        return None


def normalize_symbol(value: str) -> str:
    return value.split("@", 1)[0].strip().upper()


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"-", "--"}:
            return None
        value = text.replace(",", "")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def safe_int(value: Any) -> Optional[int]:
    num = safe_float(value)
    if num is None:
        return None
    return int(round(num))


def extract_first_float(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        result = safe_float(value)
        if result is not None:
            return result
    return None


def parse_any_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:  # treat as ms
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=JST)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = dateutil_parser.isoparse(text)
        except ValueError:
            try:
                dt = dateutil_parser.parse(text)
            except (ValueError, TypeError):
                return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=JST)
        return dt.astimezone(JST)
    return None


def parse_payload_ts(payload: Dict[str, Any]) -> Optional[datetime]:
    for key in (
        "TradingVolumeTime",
        "ExecutionDateTime",
        "CurrentPriceTime",
        "BidDateTime",
        "AskDateTime",
        "UpdateDateTime",
        "QuoteDateTime",
        "Time",
    ):
        dt = parse_any_datetime(payload.get(key))
        if dt is not None:
            return dt
    return None


def extract_levels(payload: Dict[str, Any], kind: str) -> List[Tuple[Optional[float], Optional[float]]]:
    assert kind in {"buy", "sell"}
    prefix = "Buy" if kind == "buy" else "Sell"
    levels: List[Tuple[Optional[float], Optional[float]]] = []

    def push_level(price: Optional[float], qty: Optional[float]) -> None:
        if price is None and qty is None:
            return
        key = (price, qty)
        if key in levels:
            return
        levels.append(key)

    for i in range(1, 11):
        node = payload.get(f"{prefix}{i}")
        if isinstance(node, dict):
            price = safe_float(node.get("Price") or node.get("BidPrice") or node.get("AskPrice"))
            qty = safe_float(node.get("Qty") or node.get("BidQty") or node.get("AskQty") or node.get("Volume"))
            push_level(price, qty)
        else:
            price = safe_float(payload.get(f"{prefix}Price{i}"))
            qty = safe_float(payload.get(f"{prefix}Qty{i}"))
            push_level(price, qty)

    price_list = payload.get(f"{prefix}Price")
    qty_list = payload.get(f"{prefix}Qty")
    if isinstance(price_list, list) or isinstance(qty_list, list):
        max_len = max(len(price_list) if isinstance(price_list, list) else 0, len(qty_list) if isinstance(qty_list, list) else 0)
        for idx in range(max_len):
            price = None
            qty = None
            if isinstance(price_list, list) and idx < len(price_list):
                price = safe_float(price_list[idx])
            if isinstance(qty_list, list) and idx < len(qty_list):
                entry = qty_list[idx]
                if isinstance(entry, dict):
                    qty = safe_float(entry.get("Qty") or entry.get("Volume"))
                    price = price or safe_float(entry.get("Price"))
                else:
                    qty = safe_float(entry)
            push_level(price, qty)

    return levels[:10]


def make_snapshot(
    t_recv: Any,
    payload_text: Any,
    requested_symbol: Optional[str],
) -> Optional[Snapshot]:
    try:
        payload = json.loads(payload_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    symbol_raw = (
        payload.get("Symbol")
        or payload.get("IssueCode")
        or payload.get("SymbolCode")
        or payload.get("SymbolName")
    )
    symbol = normalize_symbol(str(symbol_raw)) if symbol_raw else (requested_symbol or "")
    if requested_symbol and symbol and symbol != requested_symbol:
        return None

    symbol_name_raw = payload.get("SymbolName") or payload.get("IssueName")
    symbol_name = str(symbol_name_raw).strip() if symbol_name_raw else None

    open_price = extract_first_float(
        payload,
        (
            "OpenPrice",
            "OpeningPrice",
            "DailyOpeningPrice",
            "Open",
            "FirstPrice",
            "firstPrice",
        ),
    )
    high_price = extract_first_float(
        payload,
        (
            "HighPrice",
            "DailyHighPrice",
            "High",
            "highPrice",
        ),
    )
    low_price = extract_first_float(
        payload,
        (
            "LowPrice",
            "DailyLowPrice",
            "Low",
            "lowPrice",
        ),
    )
    close_price = extract_first_float(
        payload,
        (
            "ClosePrice",
            "SettlementPrice",
            "CurrentPrice",
            "Price",
            "LastTradedPrice",
            "Close",
            "closePrice",
        ),
    )

    cum_volume = extract_first_float(
        payload,
        (
            "TradingVolume",
            "CurrentVolume",
            "Volume",
            "SalesVolume",
            "AccumulatedVolume",
            "TotalVolume",
        ),
    )

    bid = safe_float(payload.get("BidPrice"))
    ask = safe_float(payload.get("AskPrice"))
    last = safe_float(payload.get("CurrentPrice") or payload.get("Price"))
    if last is None:
        last = close_price

    buy_levels = extract_levels(payload, "buy")
    sell_levels = extract_levels(payload, "sell")
    if bid is None and buy_levels:
        bid = buy_levels[0][0]
    if ask is None and sell_levels:
        ask = sell_levels[0][0]

    has_price = any(
        value is not None
        for value in (bid, ask, last, open_price, high_price, low_price, close_price)
    )
    if not has_price and not buy_levels and not sell_levels:
        return None

    try:
        recv_dt = datetime.fromtimestamp(float(t_recv), tz=JST)
    except Exception:
        return None

    volume_time = parse_any_datetime(payload.get("TradingVolumeTime"))

    over_sell = safe_float(payload.get("OverSellQty") or payload.get("OverSellVolume"))
    under_buy = safe_float(payload.get("UnderBuyQty") or payload.get("UnderBuyVolume"))
    total_bid = safe_float(payload.get("TotalBidQty") or payload.get("BuyTotalQty"))
    total_ask = safe_float(payload.get("TotalAskQty") or payload.get("SellTotalQty"))
    seq = safe_int(payload.get("SeqNum") or payload.get("SequenceNumber"))

    event_dt = parse_payload_ts(payload)

    return Snapshot(
        symbol=symbol or (requested_symbol or ""),
        t_recv=float(t_recv),
        recv_dt=recv_dt,
        event_dt=event_dt,
        symbol_name=symbol_name,
        volume_time=volume_time,
        bid=bid,
        ask=ask,
        last=last,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        cum_volume=cum_volume,
        buy_levels=buy_levels,
        sell_levels=sell_levels,
        over_sell=over_sell,
        under_buy=under_buy,
        total_bid=total_bid,
        total_ask=total_ask,
        seq=seq,
        payload=payload,
    )


def _effective_dt(snapshot: Snapshot) -> datetime:
    base = snapshot.volume_time or snapshot.event_dt or snapshot.recv_dt
    if base.tzinfo is None:
        return base.replace(tzinfo=JST)
    return base.astimezone(JST)


def _floor_to_interval(dt: datetime, minutes: int) -> datetime:
    local = dt.astimezone(JST)
    minute = (local.minute // minutes) * minutes
    return local.replace(minute=minute, second=0, microsecond=0)


def _price_from_snapshot(snapshot: Snapshot) -> Optional[float]:
    for value in (
        snapshot.last,
        snapshot.close_price,
        snapshot.open_price,
        snapshot.bid,
        snapshot.ask,
    ):
        if value is not None:
            return value
    return None


def _finalize_candle(state: Dict[str, Any], interval_minutes: int) -> Snapshot:
    last_snap: Snapshot = state["last"]
    return Snapshot(
        symbol=last_snap.symbol,
        t_recv=last_snap.t_recv,
        recv_dt=last_snap.recv_dt,
        event_dt=state["end"],
        symbol_name=last_snap.symbol_name,
        interval_minutes=interval_minutes,
        bar_start=state["start"],
        bar_end=state["end"],
        volume_time=state["end"],
        bid=last_snap.bid,
        ask=last_snap.ask,
        last=state["close"],
        open_price=state["open"],
        high_price=state["high"],
        low_price=state["low"],
        close_price=state["close"],
        volume=float(state["volume"]),
        cum_volume=last_snap.cum_volume,
        buy_levels=last_snap.buy_levels,
        sell_levels=last_snap.sell_levels,
        over_sell=last_snap.over_sell,
        under_buy=last_snap.under_buy,
        total_bid=last_snap.total_bid,
        total_ask=last_snap.total_ask,
        seq=last_snap.seq,
        payload=last_snap.payload,
    )


def aggregate_snapshots(raw_snaps: Sequence[Snapshot], interval_minutes: int) -> List[Snapshot]:
    if not raw_snaps:
        return []
    sorted_snaps = sorted(raw_snaps, key=_effective_dt)
    candles: List[Snapshot] = []
    current: Optional[Dict[str, Any]] = None
    last_cum: Optional[float] = None
    last_day: Optional[date] = None

    for snap in sorted_snaps:
        event_dt = _effective_dt(snap)
        price = _price_from_snapshot(snap)
        if price is None:
            continue

        day = event_dt.date()
        cum = snap.cum_volume
        if day != last_day:
            last_cum = None
        elif last_cum is not None and cum is not None and cum < last_cum - 1e-6:
            last_cum = None

        delta_volume = 0.0
        if cum is not None:
            if last_cum is not None:
                delta_volume = max(cum - last_cum, 0.0)
            last_cum = cum
        last_day = day

        bucket_start = _floor_to_interval(event_dt, interval_minutes)
        bucket_end = bucket_start + timedelta(minutes=interval_minutes)

        if current is None or bucket_start != current["start"]:
            if current is not None:
                candles.append(_finalize_candle(current, interval_minutes))
            current = {
                "start": bucket_start,
                "end": bucket_end,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": delta_volume,
                "last": snap,
            }
        else:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += delta_volume
            current["last"] = snap

    if current is not None:
        candles.append(_finalize_candle(current, interval_minutes))
    return candles


def fetch_snapshots(db_path: Path, symbol: str, limit: int, interval_minutes: int) -> List[Snapshot]:
    snapshots: List[Snapshot] = []
    query_limit = limit if limit > 0 else None
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        params: List[Any] = []
        sql = "SELECT t_recv, payload FROM raw_push"
        if symbol:
            sql += " WHERE symbol = ?"
            params.append(symbol)
        sql += " ORDER BY t_recv ASC"
        if query_limit is not None:
            sql += " LIMIT ?"
            params.append(query_limit)
        rows = conn.execute(sql, params).fetchall()

        if not rows and symbol:
            fallback_sql = "SELECT t_recv, payload FROM raw_push ORDER BY t_recv ASC"
            if query_limit is not None:
                fallback_sql += " LIMIT ?"
                rows = conn.execute(fallback_sql, (query_limit,)).fetchall()
            else:
                rows = conn.execute(fallback_sql).fetchall()

    for row in rows:
        snapshot = make_snapshot(row["t_recv"], row["payload"], symbol)
        if snapshot is None:
            continue
        if symbol and snapshot.symbol and snapshot.symbol != symbol:
            continue
        snapshots.append(snapshot)
        if query_limit is not None and len(snapshots) >= query_limit:
            break

    return aggregate_snapshots(snapshots, interval_minutes)


def fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(num) or math.isinf(num):
        return "-"
    if abs(num) >= 1000:
        return f"{num:,.0f}"
    if abs(num) >= 100:
        return f"{num:,.1f}"
    if abs(num) >= 10:
        return f"{num:,.2f}"
    return f"{num:,.3f}"


def fmt_qty(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(num) or math.isinf(num):
        return "-"
    if abs(num - round(num)) < 1e-3:
        return f"{int(round(num)):,}"
    return f"{num:,.1f}"


def summarize_snapshot(snapshot: Snapshot) -> str:
    dt_display = snapshot.primary_dt_naive()
    recv_display = snapshot.recv_dt_naive()
    symbol_label = snapshot.symbol
    if snapshot.symbol_name and snapshot.symbol_name.lower() != snapshot.symbol.lower():
        symbol_label = f"{snapshot.symbol} ({snapshot.symbol_name})"
    lines = [
        f"{symbol_label} {dt_display.strftime('%H:%M:%S.%f')[:-3]} JST",
    ]
    if snapshot.event_dt is not None:
        delta = snapshot.event_dt - snapshot.recv_dt
        if abs(delta.total_seconds()) > 0.2:
            lines.append(f"recv {recv_display.strftime('%H:%M:%S.%f')[:-3]} JST")

    if snapshot.interval_minutes:
        lines.append(f"Interval {snapshot.interval_minutes} min")
    if snapshot.bar_start and snapshot.bar_end:
        start_local = snapshot.bar_start.astimezone(JST) if snapshot.bar_start.tzinfo else snapshot.bar_start.replace(
            tzinfo=JST
        )
        end_local = snapshot.bar_end.astimezone(JST) if snapshot.bar_end.tzinfo else snapshot.bar_end.replace(tzinfo=JST)
        lines.append(
            f"Range {start_local.strftime('%H:%M:%S')} - {end_local.strftime('%H:%M:%S')}"
        )

    if snapshot.has_ohlc():
        lines.append(
            f"OHLC {fmt_price(snapshot.open_price)} / {fmt_price(snapshot.high_price)} / "
            f"{fmt_price(snapshot.low_price)} / {fmt_price(snapshot.close_price)}"
        )
    if snapshot.volume is not None:
        lines.append(f"Volume {fmt_qty(snapshot.volume)}")

    lines.append(f"Bid1 {fmt_price(snapshot.bid)} / {fmt_qty(snapshot.bid_qty())}")
    lines.append(f"Ask1 {fmt_price(snapshot.ask)} / {fmt_qty(snapshot.ask_qty())}")
    if snapshot.last is not None:
        lines.append(f"Last {fmt_price(snapshot.last)}")

    buy_top3 = sum(level[1] or 0.0 for level in snapshot.buy_levels[:3])
    sell_top3 = sum(level[1] or 0.0 for level in snapshot.sell_levels[:3])
    if buy_top3 or sell_top3:
        lines.append(f"Top3 Qty  Buy {fmt_qty(buy_top3)} / Sell {fmt_qty(sell_top3)}")

    if snapshot.under_buy is not None or snapshot.over_sell is not None:
        lines.append(
            f"UnderBuy {fmt_qty(snapshot.under_buy)} / OverSell {fmt_qty(snapshot.over_sell)}"
        )
    if snapshot.total_bid is not None or snapshot.total_ask is not None:
        lines.append(
            f"Total Qty  Bid {fmt_qty(snapshot.total_bid)} / Ask {fmt_qty(snapshot.total_ask)}"
        )
    if snapshot.seq is not None:
        lines.append(f"SeqNum {snapshot.seq}")

    depth = max(len(snapshot.buy_levels), len(snapshot.sell_levels))
    if depth:
        depth_to_show = min(depth, 5)
        lines.append("")
        lines.append("Depth (price / qty)")
        lines.append("  Bid                 | Ask")
        for idx in range(depth_to_show):
            b_price, b_qty = (
                snapshot.buy_levels[idx] if idx < len(snapshot.buy_levels) else (None, None)
            )
            s_price, s_qty = (
                snapshot.sell_levels[idx] if idx < len(snapshot.sell_levels) else (None, None)
            )
            lines.append(
                f"B{idx+1}: {fmt_price(b_price):>10} / {fmt_qty(b_qty):>10}"
                f" | S{idx+1}: {fmt_price(s_price):>10} / {fmt_qty(s_qty):>10}"
            )

    return "\n".join(lines)


def create_plot(symbol: str, snapshots: Sequence[Snapshot]) -> None:
    cand_snapshots = [snap for snap in snapshots if snap.has_ohlc()]
    if not cand_snapshots:
        print("[ERROR] OHLC を含むレコードが不足しています。ローソク足を描画できません。", file=sys.stderr)
        sys.exit(1)

    times_num = np.array(
        [mdates.date2num(snap.primary_dt_naive()) for snap in cand_snapshots], dtype=float
    )
    open_values = np.array(
        [snap.open_price if snap.open_price is not None else np.nan for snap in cand_snapshots],
        dtype=float,
    )
    high_values = np.array(
        [snap.high_price if snap.high_price is not None else np.nan for snap in cand_snapshots],
        dtype=float,
    )
    low_values = np.array(
        [snap.low_price if snap.low_price is not None else np.nan for snap in cand_snapshots],
        dtype=float,
    )
    close_values = np.array(
        [snap.close_price if snap.close_price is not None else np.nan for snap in cand_snapshots],
        dtype=float,
    )
    volume_values = np.array(
        [snap.volume if snap.volume is not None else 0.0 for snap in cand_snapshots],
        dtype=float,
    )

    unique_times = np.unique(times_num)
    if unique_times.size >= 2:
        deltas = np.diff(unique_times)
        positive = deltas[deltas > 0]
        step = float(np.min(positive)) if positive.size > 0 else 1.0 / (24 * 60)
    else:
        step = 1.0 / (24 * 60)
    candle_width = step * 0.7
    bar_width = candle_width * 0.6

    price_range = float(np.nanmax(high_values) - np.nanmin(low_values)) if high_values.size else 0.0
    min_body = price_range * 0.002 if price_range > 0 else 1e-3

    color_up = "#2ca02c"
    color_down = "#d62728"

    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(19.2, 10.8),
        dpi=100,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.subplots_adjust(right=0.72, hspace=0.05)

    first_snap = cand_snapshots[0]
    symbol_label = symbol
    if first_snap.symbol_name and first_snap.symbol_name.lower() != symbol.lower():
        symbol_label = f"{symbol} ({first_snap.symbol_name})"
    interval_label = first_snap.interval_minutes or 1

    for idx, x_val in enumerate(times_num):
        open_val = open_values[idx]
        high_val = high_values[idx]
        low_val = low_values[idx]
        close_val = close_values[idx]
        if not np.isfinite([open_val, high_val, low_val, close_val]).all():
            continue

        is_up = close_val >= open_val
        color = color_up if is_up else color_down

        ax_price.vlines(x_val, low_val, high_val, color=color, linewidth=1.0)

        body_height = abs(close_val - open_val)
        if body_height < min_body:
            rect_height = min_body
            rect_base = open_val - rect_height / 2.0
        else:
            rect_height = body_height
            rect_base = min(open_val, close_val)

        rect = Rectangle(
            (x_val - candle_width / 2.0, rect_base),
            candle_width,
            rect_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.6,
        )
        ax_price.add_patch(rect)

        ax_vol.bar(
            x_val,
            volume_values[idx],
            width=bar_width,
            color=color,
            align="center",
            alpha=0.7,
        )

    ax_price.set_title(f"{symbol_label} raw_push {interval_label}-min OHLC")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.set_xlim(times_num[0] - candle_width, times_num[-1] + candle_width)
    ax_price.xaxis_date()
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    max_volume = float(np.max(volume_values)) if volume_values.size else 0.0
    ax_vol.set_ylabel("Volume")
    ax_vol.set_xlabel("Time (JST)")
    ax_vol.grid(True, linestyle="--", alpha=0.3)
    ax_vol.set_ylim(0.0, max_volume * 1.2 if max_volume > 0 else 1.0)

    fig.autofmt_xdate()

    info_box = fig.text(
        0.78,
        0.5,
        "",
        ha="left",
        va="center",
        fontsize=10,
        family="monospace",
    )
    info_box.set_visible(False)

    cursor_price = ax_price.axvline(times_num[0], color="gray", linestyle="--", alpha=0.4)
    cursor_vol = ax_vol.axvline(times_num[0], color="gray", linestyle="--", alpha=0.4)
    cursor_price.set_visible(False)
    cursor_vol.set_visible(False)

    last_index: Dict[str, Optional[int]] = {"value": None}

    def clear_annotation() -> None:
        if last_index["value"] is None:
            return
        last_index["value"] = None
        info_box.set_visible(False)
        cursor_price.set_visible(False)
        cursor_vol.set_visible(False)
        fig.canvas.draw_idle()

    def update_annotation(index: int) -> None:
        if index < 0 or index >= len(cand_snapshots):
            return
        summary = summarize_snapshot(cand_snapshots[index])
        info_box.set_text(summary)
        info_box.set_visible(True)
        x_val = times_num[index]
        cursor_price.set_xdata([x_val, x_val])
        cursor_price.set_ydata(ax_price.get_ylim())
        cursor_price.set_visible(True)
        cursor_vol.set_xdata([x_val, x_val])
        cursor_vol.set_ydata(ax_vol.get_ylim())
        cursor_vol.set_visible(True)
        fig.canvas.draw_idle()

    def on_move(event) -> None:
        if event.xdata is None or event.inaxes not in {ax_price, ax_vol}:
            clear_annotation()
            return
        idx = int(np.searchsorted(times_num, event.xdata))
        if idx >= len(times_num):
            idx = len(times_num) - 1
        elif idx > 0 and abs(event.xdata - times_num[idx - 1]) < abs(event.xdata - times_num[idx]):
            idx -= 1
        if idx < 0 or idx >= len(cand_snapshots):
            clear_annotation()
            return
        if last_index["value"] == idx:
            return
        last_index["value"] = idx
        update_annotation(idx)

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("figure_leave_event", lambda _evt: clear_annotation())

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="raw_push の板データをチャート表示するビューア",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("db_path", help="リフィード DB (naut_market_*.db)")
    parser.add_argument("symbol", help="銘柄コード（raw_push.symbol と一致させる）")
    parser.add_argument(
        "--limit",
        type=int,
        default=4000,
        help="読み込む raw_push 行数の上限。0 で無制限（時間がかかる場合あり）。",
    )
    parser.add_argument(
        "--interval",
        type=int,
        choices=(1, 3, 5),
        default=1,
        help="足の長さ（分）。1/3/5 分足を切り替え可能。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        print(f"[ERROR] DB が見つかりません: {db_path}", file=sys.stderr)
        sys.exit(2)

    symbol = normalize_symbol(args.symbol)
    snapshots = fetch_snapshots(db_path, symbol, args.limit, args.interval)
    if not snapshots:
        print(f"[ERROR] 対象シンボルの板データが raw_push に存在しません: {symbol}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] aggregated {len(snapshots)} bars ({args.interval}-min) for {symbol}")
    create_plot(symbol, snapshots)


if __name__ == "__main__":
    main()
