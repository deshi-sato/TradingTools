# -*- coding: utf-8 -*-
from pathlib import Path
text = Path("scripts/stream_microbatch.py").read_text(encoding="utf-8")
targets = set("縺蜑譛繧逕譎譚萓險閾螳逶譟霑蜿蟆謖諱牁冁孁宁紁刁嗚ぁぁなぁ�E")
for idx, line in enumerate(text.splitlines(), 1):
    if any(ch in targets for ch in line):
        print(f"{idx:4}: {line}")

