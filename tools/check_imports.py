import sys

mods = ["openpyxl", "pandas", "xlwings"]
for name in mods:
    try:
        m = __import__(name)
        print(f"OK {name} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"NG {name} {e.__class__.__name__}: {e}")
        sys.exit(2)

