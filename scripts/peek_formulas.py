\"\"\"peek_formulas.py : Auto-generated placeholder

- file: scripts/peek_formulas.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import sys


TARGET_CELLS = {"A1", "AA1"}


def _ns(tag):
    if tag.startswith("{"):
        return tag[1:].split("}")[0]
    return ""


def show_formulas(xlsm_path: str):
    print(f"== {xlsm_path} ==")
    with ZipFile(xlsm_path, "r") as z:
        for name in z.namelist():
            if not name.startswith("xl/worksheets/") or not name.endswith(".xml"):
                continue
            try:
                data = z.read(name)
                root = ET.fromstring(data)
            except Exception:
                continue
            ns = _ns(root.tag)
            c_tag = f"{{{ns}}}c" if ns else "c"
            f_tag = f"{{{ns}}}f" if ns else "f"
            for c in root.iter(c_tag):
                r = c.attrib.get("r")
                if r not in TARGET_CELLS:
                    continue
                f = c.find(f_tag)
                if f is not None:
                    print(name, r, f.text)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        show_formulas(p)

