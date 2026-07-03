"""pywebview shell for the pocketmidi GUI.

This module owns the window and the js_api bridge ONLY. All engine access lives in
adapter.py, which must stay pywebview-free — it is the seam a future thin web-demo
server would wrap. On the front-end side, bridge.js is the only file that knows
about window.pywebview.
"""

from __future__ import annotations

from pathlib import Path

import webview

WEB_DIR = Path(__file__).parent / "web"

WINDOW_TITLE = "pocketmidi"


class Api:
    """js_api surface exposed to the webview.

    Phase 0: connectivity stub only. Phase 1 delegates these to adapter.Session.
    """

    def ping(self) -> str:
        return "pong"


def main() -> None:
    api = Api()
    webview.create_window(
        WINDOW_TITLE,
        str(WEB_DIR / "index.html"),
        js_api=api,
        width=1120,
        height=780,
        min_size=(940, 660),
        background_color="#141416",
    )
    webview.start()


if __name__ == "__main__":
    main()
