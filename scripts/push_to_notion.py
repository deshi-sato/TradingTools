#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post a summary entry to a Notion database with Japanese properties:

- 日付 (Date)
- 目的 (Text/Rich text)
- 実行内容 (Text/Rich text)
- 気づき・課題 (Text/Rich text)
- 次回アクション (Text/Rich text)
- タグ (Multi-select)

Accepts either a Database ID or a Page ID that contains a child database.
If a Page ID is provided, the script searches for the first child_database block
and posts to that database.

Usage:
  set NOTION_TOKEN=secret_xxx
  python scripts/push_to_notion.py \
    --db-id 25fc4954305e80cd898fe9544f59b205 \
    --title "シグナル最適化ログ" \
    --date 2025-09-05 \
    --purpose "DB保持期間拡張と最適化/WF構築" \
    --execution "変更点・実行手順..." \
    --findings "結果・勝率..." \
    --next-actions "次のアクション..." \
    --tags DB,Optimization,Walkforward,Notion
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any, Optional
import json
from urllib import request, error


NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push a summary row to a Notion database")
    p.add_argument("--db-id", required=True, help="Notion database ID or page ID (32-char ID)")
    p.add_argument("--title", required=True, help="Entry title")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--purpose", required=True)
    p.add_argument("--execution", required=True)
    p.add_argument("--findings", required=True)
    p.add_argument("--next-actions", required=True)
    p.add_argument("--tags", default="", help="Comma separated tags for multi-select")
    p.add_argument("--token", default=None, help="Override NOTION_TOKEN env var (optional)")
    return p.parse_args()


def notion_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def notion_request(method: str, path: str, token: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{NOTION_API}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method=method)
    for k, v in notion_headers(token).items():
        req.add_header(k, v)
    try:
        with request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {body}")


def get_db_schema(token: str, db_id: str) -> Dict[str, Any]:
    return notion_request("GET", f"/databases/{db_id}", token)


def find_title_prop_name(db_json: Dict[str, Any]) -> str:
    props = db_json.get("properties", {})
    for name, meta in props.items():
        if meta.get("type") == "title":
            return name
    return "Name"


def to_rich_text(text: str) -> Dict[str, Any]:
    return {"rich_text": [{"type": "text", "text": {"content": text or ""}}]}


def resolve_child_database(token: str, page_id: str) -> Optional[str]:
    data = notion_request("GET", f"/blocks/{page_id}/children", token)
    for b in data.get("results", []):
        if b.get("type") == "child_database":
            return b.get("id")
    return None


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("NOTION_TOKEN")
    if not token:
        print("[ERROR] NOTION_TOKEN environment variable not set.")
        sys.exit(2)

    # Try database directly; if ID is a page, try to resolve a child database under it.
    try:
        db = get_db_schema(token, args.db_id)
    except Exception as e:
        msg = str(e)
        if ("is a page, not a database" in msg) or ("Provided ID" in msg and "is a page" in msg):
            alt = resolve_child_database(token, args.db_id)
            if not alt:
                print("[ERROR] Provided ID is a page and no child database was found under it.")
                print("Open the database as a full page and pass its ID, or place a database block in the page and retry.")
                print(f"Original error: {msg}")
                sys.exit(3)
            try:
                db = get_db_schema(token, alt)
                args.db_id = alt
            except Exception as e2:
                print(f"[ERROR] Failed to fetch resolved child database schema: {e2}")
                sys.exit(3)
        else:
            print(f"[ERROR] Failed to fetch Notion DB schema: {msg}")
            sys.exit(3)

    title_prop = find_title_prop_name(db)

    props: Dict[str, Any] = {}
    props[title_prop] = {"title": [{"type": "text", "text": {"content": args.title}}]}
    props["日付"] = {"date": {"start": args.date}}
    props["目的"] = to_rich_text(args.purpose)
    props["実行内容"] = to_rich_text(args.execution)
    props["気づき・課題"] = to_rich_text(args.findings)
    props["次回アクション"] = to_rich_text(args.next_actions)

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    if tags:
        props["タグ"] = {"multi_select": [{"name": t} for t in tags]}

    payload = {"parent": {"database_id": args.db_id}, "properties": props}

    try:
        _ = notion_request("POST", "/pages", token, payload)
    except Exception as e:
        print(f"[ERROR] Failed to create Notion page: {e}")
        sys.exit(4)

    print("[OK] Notion page created.")


if __name__ == "__main__":
    main()

