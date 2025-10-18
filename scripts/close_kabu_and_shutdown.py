#!/usr/bin/env python
# close_kabu_and_shutdown.py
import subprocess
import sys
import time

import psutil
from pywinauto import Application, Desktop
from pywinauto.timings import TimeoutError

KABU_PROCESS_NAMES = {"kabu_station.exe", "kabuステーション.exe"}
CONFIRM_DIALOG_TITLE = "終了してよろしいですか"
YES_TITLES = {"はい", "OK"}
FALLBACK_WAIT_SEC = 5
SHUTDOWN_DELAY_SEC = 3


def find_kabu_process() -> psutil.Process | None:
    for proc in psutil.process_iter(["pid", "name"]):
        name = (proc.info.get("name") or "").lower()
        if name and name in {n.lower() for n in KABU_PROCESS_NAMES}:
            return proc
    return None


def close_kabu(proc: psutil.Process) -> None:
    try:
        app = Application(backend="uia").connect(process=proc.pid, timeout=5)
    except Exception as exc:
        print(f"[WARN] UI automation attach failed ({exc}); terminating the process.")
        _terminate_process(proc)
        return

    main_window = app.top_window()
    main_window.set_focus()
    main_window.close()

    desktop = Desktop(backend="uia")
    try:
        dialog = desktop.window(title=CONFIRM_DIALOG_TITLE)
        dialog.wait("exists ready", timeout=FALLBACK_WAIT_SEC)
        for title in YES_TITLES:
            yes_button = dialog.child_window(title=title, control_type="Button")
            if yes_button.exists():
                yes_button.click_input()
                break
        else:
            dialog.type_keys("%Y")  # Alt+Y fallback
    except TimeoutError:
        pass
    except Exception as exc:
        print(f"[WARN] Confirmation dialog handling failed ({exc}); killing process.")
        _terminate_process(proc)

    _wait_for_exit(proc)


def _terminate_process(proc: psutil.Process) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def _wait_for_exit(proc: psutil.Process) -> None:
    try:
        proc.wait(timeout=10)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        _terminate_process(proc)


def shutdown_windows() -> None:
    time.sleep(SHUTDOWN_DELAY_SEC)
    subprocess.run(["shutdown", "/s", "/t", "0"], check=True)


def main() -> None:
    proc = find_kabu_process()
    if proc:
        print("[INFO] Found kabuステーション; closing…")
        close_kabu(proc)
    else:
        print("[INFO] kabuステーション is not running.")
    shutdown_windows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Aborted by user.")
        sys.exit(1)
