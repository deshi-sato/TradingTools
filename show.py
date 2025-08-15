import sys
from pathlib import Path
path = sys.argv[1]
encodings = ['utf-8', 'cp932', 'shift_jis', 'utf-8-sig']
for enc in encodings:
    try:
        with open(path, 'r', encoding=enc) as f:
            lines = f.readlines()
        break
    except Exception:
        continue
else:
    with open(path, 'rb') as f:
        data = f.read().splitlines()
    lines = [l.decode('utf-8', errors='ignore') for l in data]

out = ''.join(f"{i:04d}: {line}" for i, line in enumerate(lines, 1))
Path('show_out.txt').write_text(out, encoding='utf-8')
print('WROTE show_out.txt')
