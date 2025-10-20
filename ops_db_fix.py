#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply ops DB migration to ensure paper_pairs(order_id) is UNIQUE,
so that INSERT ... ON CONFLICT(order_id) works without sqlite errors.
Usage:
  py ops_db_fix.py path\to\naut_ops.db
"""
import sys, sqlite3

def main():
    if len(sys.argv) < 2:
        print("Usage: py ops_db_fix.py <path_to_ops_db>", file=sys.stderr)
        sys.exit(2)
    db_path = sys.argv[1]
    sql = open('ops_add_unique_indexes.sql', 'r', encoding='utf-8').read()
    con = sqlite3.connect(db_path)
    try:
        con.executescript(sql)
        con.commit()
        print("Applied: UNIQUE index on paper_pairs(order_id)")
    finally:
        con.close()

if __name__ == "__main__":
    main()
