# -*- coding: utf-8 -*-
from pathlib import Path
raw = Path("scripts/stream_microbatch.py").read_bytes()
encodings = ["utf-8", "cp932", "shift_jis", "euc-jp"]
for enc in encodings:
    try:
        txt = raw.decode(enc)
    except Exception as e:
        print(enc, "-> error", e)
    else:
        first = txt.splitlines()[3] if len(txt.splitlines()) > 3 else txt
        print(enc, "->", first)

