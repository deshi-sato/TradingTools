import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo
import xml.etree.ElementTree as ET


TARGET_CELLS = {"A1", "AA1"}
# Remove a trailing suffix like "=> 関数使用上限に達しました。" (period optional, whitespace flexible)
SUFFIX_RE = re.compile(r"[\s]*=>\s*関数使用上限に達しました。?$")


def _ns(tag):
    # Helper to extract namespace prefix from a tag like '{ns}worksheet'
    if tag.startswith("{"):
        return tag[1:].split("}")[0]
    return ""


def clean_sheet_xml(data: bytes) -> bytes:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return data  # leave untouched if parse fails

    ns = _ns(root.tag)
    if ns:
        c_tag = f"{{{ns}}}c"
        f_tag = f"{{{ns}}}f"
    else:
        c_tag = "c"
        f_tag = "f"

    changed = False
    for c in root.iter(c_tag):
        r = c.attrib.get("r")
        if r not in TARGET_CELLS:
            continue
        f = c.find(f_tag)
        if f is None or f.text is None:
            continue
        new_text = SUFFIX_RE.sub("", f.text)
        if new_text != f.text:
            f.text = new_text
            changed = True

    if not changed:
        return data

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def process_file(path: Path) -> None:
    if not path.exists():
        print(f"skip (missing): {path}")
        return

    backup = path.with_suffix(path.suffix + ".bak_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    shutil.copy2(path, backup)
    print(f"backup: {backup}")

    with tempfile.TemporaryDirectory() as td:
        tmp_zip_path = Path(td) / (path.name + ".tmp")
        with ZipFile(path, "r") as zin, ZipFile(tmp_zip_path, "w", compression=ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename.startswith("xl/worksheets/") and info.filename.endswith(".xml"):
                    data = clean_sheet_xml(data)

                zi = ZipInfo(info.filename)
                # Preserve timestamps and attributes
                zi.date_time = info.date_time
                zi.compress_type = ZIP_DEFLATED
                zi.external_attr = info.external_attr
                zi.create_system = info.create_system
                zi.flag_bits = info.flag_bits
                zout.writestr(zi, data)

        # Replace original safely
        shutil.move(tmp_zip_path, path)
        print(f"fixed: {path}")


def main(argv):
    if len(argv) < 2:
        print("Usage: python scripts/fix_excel_formulas.py <file1.xlsm> [file2.xlsm ...]")
        return 2
    for p in argv[1:]:
        process_file(Path(p))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

