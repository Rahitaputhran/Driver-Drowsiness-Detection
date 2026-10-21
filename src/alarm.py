from __future__ import annotations

import platform


def beep():
    """Best-effort desktop alarm. Does not crash the inference loop."""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1400, 250)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass
