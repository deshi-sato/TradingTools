#!/usr/bin/env python
# -*- coding: utf-8 -*-
# kabuステーションを正常終了（確認ダイアログ自動YES）→ Windowsをシャットダウン
import ctypes
import ctypes.wintypes as wt
import time
import subprocess
import sys
import psutil

user32 = ctypes.windll.user32

# ---- 互換: ULONG_PTR が無い環境対策 -----------------------------------------
try:
    ULONG_PTR = wt.ULONG_PTR  # ある環境（Py3.11等）では用意されている
except AttributeError:
    # ポインタと同じ幅の整数型を自前定義（32bit: unsigned long / 64bit: unsigned long long）
    ULONG_PTR = (
        ctypes.c_ulong if ctypes.sizeof(ctypes.c_void_p) == 4 else ctypes.c_ulonglong
    )

# --- WinAPI prototypes -------------------------------------------------------
EnumWindows = user32.EnumWindows
EnumChildWindows = user32.EnumChildWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
GetWindowThreadProcessId = user32.GetWindowThreadProcessId
IsWindowVisible = user32.IsWindowVisible
GetWindowTextW = user32.GetWindowTextW
GetWindowTextLengthW = user32.GetWindowTextLengthW
GetClassNameW = user32.GetClassNameW
PostMessageW = user32.PostMessageW
SendMessageW = user32.SendMessageW
GetDlgItem = user32.GetDlgItem
FindWindowW = user32.FindWindowW
SetForegroundWindow = user32.SetForegroundWindow
SendInput = user32.SendInput

# --- constants ---------------------------------------------------------------
WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
BM_CLICK = 0x00F5
IDYES = 6
VK_MENU = 0x12  # Alt
VK_Y = 0x59

DIALOG_CLASS = "#32770"  # 標準ダイアログのクラス
POST_CLOSE_WAIT_SEC = 0.8
DIALOG_WAIT_SEC = 12
EXIT_WAIT_SEC = 15
SHUTDOWN_DELAY_SEC = 2

PROCESS_NAME_HINTS = ("kabu", "ステーション")


# --- helpers -----------------------------------------------------------------
def _get_text(hwnd: int) -> str:
    length = GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _get_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    GetClassNameW(hwnd, buf, 256)
    return buf.value


def _enum_top_windows() -> list[int]:
    hwnds = []

    def _cb(hwnd, lParam):
        if IsWindowVisible(hwnd):
            hwnds.append(hwnd)
        return True

    EnumWindows(EnumWindowsProc(_cb), 0)
    return hwnds


def _enum_child_windows(parent: int) -> list[int]:
    childs = []

    def _cb(hwnd, lParam):
        childs.append(hwnd)
        return True

    EnumChildWindows(parent, EnumWindowsProc(_cb), 0)
    return childs


def _find_kabu_process() -> psutil.Process | None:
    for p in psutil.process_iter(["pid", "name"]):
        name = (p.info.get("name") or "").lower()
        if "kabu" in name:
            return p
    return None


def _main_window_for_pid(pid: int) -> int | None:
    # 可視トップレベルでタイトルにヒント語を含むものを優先
    for hwnd in _enum_top_windows():
        txt = _get_text(hwnd)
        if not txt:
            continue
        proc_id = wt.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
        if proc_id.value != pid:
            continue
        if any(hint in txt for hint in PROCESS_NAME_HINTS):
            return hwnd
    # だめなら同PIDの可視ウィンドウの先頭
    for hwnd in _enum_top_windows():
        proc_id = wt.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
        if proc_id.value == pid:
            return hwnd
    return None


def _click_yes_by_controls(hDlg: int) -> bool:
    # 1) IDYES(6) を直接クリック
    hYes = GetDlgItem(hDlg, IDYES)
    if hYes:
        SendMessageW(hYes, BM_CLICK, 0, 0)
        return True
    # 2) 子ウィンドウ列挙して Button の「はい/Yes」をクリック
    for c in _enum_child_windows(hDlg):
        if _get_class(c).lower() == "button":
            text = _get_text(c)
            if text and ("はい" in text or "Yes" in text or "YES" in text):
                SendMessageW(c, BM_CLICK, 0, 0)
                return True
    return False


# SendInput 用の構造体
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    _anonymous_ = ("i",)
    _fields_ = [("type", wt.DWORD), ("i", _I)]


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002


def _send_alt_y(hWnd: int) -> None:
    # フォールバック：前面化して Alt+Y を送信
    try:
        SetForegroundWindow(hWnd)
    except Exception:
        pass
    events = (INPUT * 4)()
    # Alt down
    events[0].type = INPUT_KEYBOARD
    events[0].ki = KEYBDINPUT(VK_MENU, 0, 0, 0, 0)
    # 'Y' down
    events[1].type = INPUT_KEYBOARD
    events[1].ki = KEYBDINPUT(VK_Y, 0, 0, 0, 0)
    # 'Y' up
    events[2].type = INPUT_KEYBOARD
    events[2].ki = KEYBDINPUT(VK_Y, 0, KEYEVENTF_KEYUP, 0, 0)
    # Alt up
    events[3].type = INPUT_KEYBOARD
    events[3].ki = KEYBDINPUT(VK_MENU, 0, KEYEVENTF_KEYUP, 0, 0)
    SendInput(4, ctypes.byref(events), ctypes.sizeof(INPUT))


def _hunt_and_accept_dialog(timeout_sec: int = DIALOG_WAIT_SEC) -> bool:
    # #32770 の全ダイアログから 'kabu/カブ/ステーション' を含むものを探し、Yesを押す
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        for hwnd in _enum_top_windows():
            if _get_class(hwnd) != DIALOG_CLASS:
                continue
            title = _get_text(hwnd)
            if not any(k in title for k in ("kabu", "カブ", "ステーション")):
                continue
            if _click_yes_by_controls(hwnd):
                return True
            _send_alt_y(hwnd)  # 最終手段
            return True
        time.sleep(0.2)
    return False


def close_kabu_and_shutdown():
    proc = _find_kabu_process()
    if proc:
        pid = proc.pid
        hwnd = _main_window_for_pid(pid)
        if hwnd:
            PostMessageW(hwnd, WM_CLOSE, 0, 0)
            time.sleep(POST_CLOSE_WAIT_SEC)
            _hunt_and_accept_dialog()
        else:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            proc.wait(timeout=EXIT_WAIT_SEC)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
    time.sleep(SHUTDOWN_DELAY_SEC)
    subprocess.run(["shutdown", "/s", "/t", "0"], check=False)


if __name__ == "__main__":
    try:
        close_kabu_and_shutdown()
    except KeyboardInterrupt:
        sys.exit(1)
