"""pywebview shell for the pocketmidi GUI.

This module owns the window, native file dialogs, and the js_api bridge ONLY.
All engine access lives in adapter.py, which must stay pywebview-free — it is
the seam a future thin web-demo server would wrap. On the front-end side,
bridge.js is the only file that knows about window.pywebview.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import webview

from pocketmidi_gui import adapter

WEB_DIR = Path(__file__).parent / "web"

WINDOW_TITLE = "pocketmidi"

MIDI_FILE_TYPES = ("MIDI files (*.mid;*.midi)", "All files (*.*)")


class Api:
    """js_api surface exposed to the webview. Thin delegation to adapter.Session.

    The Session (and its ~0.8s profile load) is built on a background thread so
    the window appears immediately; API calls block on readiness.
    """

    def __init__(self, autoload_path: str | None = None) -> None:
        self._window: webview.Window | None = None
        self._session: adapter.Session | None = None
        self._session_error: str | None = None
        self._autoload_path = autoload_path
        self._ready = threading.Event()
        threading.Thread(target=self._init_session, daemon=True).start()

    def _init_session(self) -> None:
        try:
            self._session = adapter.Session()
        except Exception as exc:
            self._session_error = f"Profile load failed: {exc}"
        finally:
            self._ready.set()

    def _get_session(self) -> adapter.Session | None:
        self._ready.wait()
        return self._session

    def cleanup(self) -> None:
        if self._session is not None:
            self._session.cleanup()

    # -- js_api methods ------------------------------------------------------

    def ping(self) -> str:
        return "pong"

    def get_status(self) -> dict:
        self._ready.wait()
        if self._session_error:
            return {"ok": False, "error": self._session_error}
        return {"ok": True}

    def autoload(self) -> dict:
        """Load the file given on the command line (pocketmidi-gui song.mid), once."""
        path, self._autoload_path = self._autoload_path, None
        if not path:
            return {"ok": False, "none": True}
        session = self._get_session()
        if session is None:
            return {"ok": False, "error": self._session_error}
        return session.load(path)

    def open_midi(self) -> dict:
        session = self._get_session()
        if session is None:
            return {"ok": False, "error": self._session_error}
        paths = self._window.create_file_dialog(
            webview.OPEN_DIALOG, file_types=MIDI_FILE_TYPES
        )
        if not paths:
            return {"ok": False, "cancelled": True}
        path = paths[0] if isinstance(paths, (list, tuple)) else paths
        return session.load(path)

    def humanise(self, params: dict) -> dict:
        session = self._get_session()
        if session is None:
            return {"ok": False, "error": self._session_error}
        return session.humanise_current(params)

    def undo(self) -> dict:
        session = self._get_session()
        if session is None:
            return {"ok": False, "error": self._session_error}
        return session.undo()

    def export_midi(self) -> dict:
        session = self._get_session()
        if session is None:
            return {"ok": False, "error": self._session_error}
        if session.render_path is None:
            return {"ok": False, "error": "Nothing to export — humanise first."}
        suggested = (
            f"{session.original_path.stem}_humanised.mid"
            if session.original_path else "humanised.mid"
        )
        dest = self._window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=suggested, file_types=MIDI_FILE_TYPES
        )
        if not dest:
            return {"ok": False, "cancelled": True}
        dest_path = dest[0] if isinstance(dest, (list, tuple)) else dest
        return session.export_to(dest_path)


def main() -> None:
    autoload = sys.argv[1] if len(sys.argv) > 1 and Path(sys.argv[1]).is_file() else None
    api = Api(autoload_path=autoload)
    window = webview.create_window(
        WINDOW_TITLE,
        str(WEB_DIR / "index.html"),
        js_api=api,
        width=1120,
        height=780,
        min_size=(940, 660),
        background_color="#141416",
    )
    api._window = window
    window.events.closed += api.cleanup
    webview.start()


if __name__ == "__main__":
    main()
