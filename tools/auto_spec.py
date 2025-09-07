import os
import sys
import ast
from typing import List, Dict, Any, Optional, Tuple


EXCLUDE_DIRS = {"tests", "test", ".git", ".github", "venv", ".venv", "__pycache__"}


def read_text_best_effort(path: str) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp932", "shift_jis", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err  # type: ignore[misc]


def list_python_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def get_docstring(node: ast.AST) -> Optional[str]:
    try:
        return ast.get_docstring(node, clean=True)
    except Exception:
        return None


def name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{name_of(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return name_of(node.value)
    return type(node).__name__


def const_str(node: Optional[ast.AST], source: str) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, (ast.Constant,)):
        return repr(node.value)
    # try to get exact source snippet
    try:
        seg = ast.get_source_segment(source, node)
        if seg:
            return seg.strip()
    except Exception:
        pass
    return None


def format_args_sig(args: ast.arguments, source: str) -> str:
    parts: List[str] = []
    pos = []
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    for a, d in zip(args.args, defaults):
        ann = const_str(a.annotation, source)
        seg = a.arg
        if ann:
            seg += f": {ann}"
        dstr = const_str(d, source)
        if dstr is not None:
            seg += f" = {dstr}"
        pos.append(seg)
    parts.extend(pos)
    if args.vararg:
        ann = const_str(args.vararg.annotation, source)
        parts.append("*" + args.vararg.arg + (f": {ann}" if ann else ""))
    elif args.kwonlyargs:
        parts.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        ann = const_str(a.annotation, source)
        seg = a.arg
        if ann:
            seg += f": {ann}"
        dstr = const_str(d, source)
        if dstr is not None:
            seg += f" = {dstr}"
        parts.append(seg)
    if args.kwarg:
        ann = const_str(args.kwarg.annotation, source)
        parts.append("**" + args.kwarg.arg + (f": {ann}" if ann else ""))
    return "(" + ", ".join(parts) + ")"


def collect_symbols(tree: ast.Module, source: str) -> Dict[str, Any]:
    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    exceptions: List[str] = []
    opens: List[Tuple[Optional[str], Optional[str]]] = []  # (filename, mode)
    envvars: List[str] = []
    argparse_opts: List[str] = []
    logging_used: List[str] = []
    sqlite_used: List[str] = []
    has_main = False
    settings: List[Tuple[str, Optional[str]]] = []  # NAME, value

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "sig": format_args_sig(node.args, source),
                    "returns": const_str(node.returns, source),
                    "doc": get_docstring(node),
                }
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "sig": format_args_sig(node.args, source),
                    "returns": const_str(node.returns, source),
                    "doc": get_docstring(node),
                    "async": True,
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods: List[Dict[str, Any]] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        {
                            "name": item.name,
                            "sig": format_args_sig(item.args, source),
                            "returns": const_str(item.returns, source),
                            "doc": get_docstring(item),
                            "async": isinstance(item, ast.AsyncFunctionDef),
                        }
                    )
            classes.append(
                {
                    "name": node.name,
                    "doc": get_docstring(node),
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.If):
            # detect __main__ guard
            try:
                test_src = ast.get_source_segment(source, node.test) or ""
            except Exception:
                test_src = ""
            if "__name__" in test_src and "__main__" in test_src:
                has_main = True
        elif isinstance(node, ast.Assign):
            # capture UPPER_CASE simple settings
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    settings.append((target.id, const_str(node.value, source)))

    # Walk full tree for calls/usages
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                if h.type is not None:
                    exceptions.append(name_of(h.type))
        if isinstance(node, ast.Call):
            func_name = name_of(node.func)
            if func_name.endswith("open") or func_name == "open":
                filename = None
                mode = None
                if node.args:
                    filename = const_str(node.args[0], source)
                for kw in node.keywords or []:
                    if kw.arg == "mode":
                        mode = const_str(kw.value, source)
                if len(node.args) > 1 and mode is None:
                    mode = const_str(node.args[1], source)
                opens.append((filename, mode))
            if func_name.startswith("logging."):
                logging_used.append(func_name)
            if func_name.startswith("sqlite3.connect"):
                dsn = None
                if node.args:
                    dsn = const_str(node.args[0], source)
                sqlite_used.append(dsn or "<dynamic>")
            if func_name.endswith("add_argument"):
                # argparse options
                parts: List[str] = []
                for a in node.args:
                    s = const_str(a, source)
                    if s:
                        parts.append(s)
                for kw in node.keywords or []:
                    if kw.arg:
                        v = const_str(kw.value, source)
                        parts.append(f"{kw.arg}={v}")
                if parts:
                    argparse_opts.append(
                        ", ".join(parts[:3]) + (" ..." if len(parts) > 3 else "")
                    )
        if isinstance(node, ast.Subscript):
            # os.environ["FOO"]
            if name_of(node.value).endswith("os.environ"):
                key = const_str(node.slice, source)
                if key:
                    envvars.append(key.strip("'\""))
        if isinstance(node, ast.Call):
            if name_of(node.func).endswith("os.getenv"):
                if node.args:
                    key = const_str(node.args[0], source)
                    if key:
                        envvars.append(key.strip("'\""))

    return {
        "functions": functions,
        "classes": classes,
        "exceptions": sorted(set(exceptions)),
        "opens": opens,
        "envvars": sorted(set(envvars)),
        "argparse": argparse_opts,
        "logging": sorted(set(logging_used)),
        "sqlite": sqlite_used,
        "has_main": has_main,
        "settings": settings,
    }


def render_markdown(rel_path: str, source: str, info: Dict[str, Any]) -> str:
    title = f"{rel_path} 仕様書"
    module_doc = get_docstring(ast.parse(source)) or ""

    def bullet_list(items: List[str]) -> str:
        return "\n".join(f"- {x}" for x in items) if items else "- なし"

    # 概要 / 目的 は推測が難しいため簡潔に
    overview = module_doc.splitlines()[0] if module_doc else "このスクリプトの高レベルな機能を記述してください。"
    purpose = "業務/運用上の目的を簡潔に記述してください。"

    # 入力/出力の推測
    inputs: List[str] = []
    outputs: List[str] = []
    for fn, mode in info["opens"]:
        if mode and any(m in mode for m in ["'w'", '"w"', "'a'", '"a"', "'wb'", '"wb"', "'ab'", '"ab"']):
            outputs.append(f"ファイル出力: {fn or '<動的パス>'} mode={mode}")
        else:
            inputs.append(f"ファイル入力: {fn or '<動的パス>'}")
    for dsn in info["sqlite"]:
        inputs.append(f"DB接続: sqlite3 {dsn}")

    if info["argparse"]:
        inputs.append("コマンドライン引数: argparse によるオプションを受け付けます")

    if info["envvars"]:
        inputs.append("環境変数: " + ", ".join(info["envvars"]))

    # 設定項目
    settings_lines = [
        f"{k}: {v if v is not None else '<動的/不明>'}" for k, v in info["settings"]
    ]

    # 処理フロー（推測ベースの骨子）
    flow: List[str] = []
    flow.append("起動: __main__ ブロック" + ("あり" if info["has_main"] else "なし"))
    if info["argparse"]:
        flow.append("引数解析: argparse でオプションを解析")
    if info["opens"]:
        flow.append("入出力: ファイルの読み書きを実施")
    if info["sqlite"]:
        flow.append("データアクセス: sqlite3 に接続・操作")
    if info["logging"]:
        flow.append("ロギング: logging による実行ログ出力")
    flow.append("コア処理: 主要関数を順次呼び出し")

    # 主要関数/クラス
    fn_lines = [
        f"{f['name']}{f['sig']} -> {f['returns'] or 'None'}: { (f['doc'] or '説明なし').splitlines()[0] }"
        for f in info["functions"]
    ]
    cls_lines: List[str] = []
    for c in info["classes"]:
        cls_lines.append(f"{c['name']}: {(c['doc'] or '説明なし').splitlines()[0]}")
        for m in c["methods"]:
            cls_lines.append(
                f"  - {m['name']}{m['sig']} -> {m['returns'] or 'None'}: {(m['doc'] or '説明なし').splitlines()[0]}"
            )

    # エラー・ログ
    err_lines = info["exceptions"]
    log_lines = sorted(set([n.split(".")[-1] for n in info["logging"]]))

    # 注意点
    notes = [
        "実行環境: Python 3.x 標準ライブラリで動作",
        "パフォーマンス: 入出力/DBアクセス量によって変動",
        "前提: 必要な入力ファイル/DBが存在すること",
    ]

    md = []
    md.append(f"# {title}")
    md.append("")
    md.append("## 概要")
    md.append(overview)
    md.append("")
    md.append("## 目的")
    md.append(purpose)
    md.append("")
    md.append("## 入力")
    md.append(bullet_list(inputs))
    md.append("")
    md.append("## 出力")
    md.append(bullet_list(outputs))
    md.append("")
    md.append("## 設定項目")
    md.append(bullet_list(settings_lines))
    md.append("")
    md.append("## 処理フロー")
    md.append(bullet_list(flow))
    md.append("")
    md.append("## 主要関数・クラス")
    if fn_lines:
        md.extend(f"- {line}" for line in fn_lines)
    else:
        md.append("- なし")
    if cls_lines:
        md.extend(f"- {line}" for line in cls_lines)
    md.append("")
    md.append("## 代表的なエラー")
    md.append(bullet_list(err_lines))
    md.append("")
    md.append("## ログ")
    if log_lines:
        md.append("- 使用箇所: logging." + ", ".join(log_lines))
    else:
        md.append("- なし")
    md.append("")
    md.append("## 注意点・制約")
    md.append(bullet_list(notes))

    return "\n".join(md) + "\n"


def write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def rel_to_spec_path(rel_path: str) -> str:
    # Mirror directory structure under docs/specs and change suffix to .md
    return os.path.join("docs", "specs", rel_path) + ".md"


def main(argv: List[str]) -> int:
    root = os.path.abspath(argv[1]) if len(argv) > 1 else os.getcwd()
    py_files = list_python_files(root)
    # ensure deterministic paths relative to root
    for path in py_files:
        rel = os.path.relpath(path, root).replace("\\", "/")
        try:
            src = read_text_best_effort(path)
            tree = ast.parse(src)
        except Exception as e:  # noqa: BLE001
            content = (
                f"# {rel} 仕様書\n\n"
                "解析エラーにより自動生成できませんでした。\n\n"
                f"- エラー: {type(e).__name__}: {e}\n"
            )
            out = rel_to_spec_path(rel)
            write_file(out, content)
            continue

        info = collect_symbols(tree, src)
        md = render_markdown(rel, src, info)
        out = rel_to_spec_path(rel)
        write_file(out, md)

    print(f"Generated specs for {len(py_files)} files under docs/specs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

