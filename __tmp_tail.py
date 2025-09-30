# -*- coding: utf-8 -*-
from pathlib import Path
text = Path("scripts/stream_microbatch.py").read_text(encoding="utf-8").splitlines()
for idx in range(150, 380):
    print(f"{idx:3}: {text[idx-1]}")

