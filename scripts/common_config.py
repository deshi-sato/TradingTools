"""共通の設定ファイル読み込みユーティリティ。UTF-8(BOM対応)で読み込み、明快なエラーを返す。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_SNIPPET_LEN = 120


def _format_snippet(text: str) -> str:
    cleaned = text.replace("\\r", " ").replace("\\n", " ")
    return cleaned[:_SNIPPET_LEN]


def _snippet_from_bytes(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            raw = fh.read(_SNIPPET_LEN * 4)
    except OSError:
        return ""
    return _format_snippet(raw.decode("utf-8", errors="replace"))


def load_json_utf8(path: str) -> Dict[str, Any]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        snippet = _snippet_from_bytes(target)
        raise RuntimeError(
            f"Failed to decode JSON config {target}: {exc}; snippet=\"{snippet}\""
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read JSON config {target}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = _format_snippet(text)
        raise RuntimeError(
            f"Failed to parse JSON config {target}: {exc}; snippet=\"{snippet}\""
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"JSON config {target} must be an object; got {type(data).__name__}"
        )
    return data
