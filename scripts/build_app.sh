#!/usr/bin/env bash
# Build the macOS .app for the wobblemidi GUI (PyInstaller; spec checked in at
# wobblemidi-gui.spec). Produces dist/wobblemidi.app and a release zip, then
# smoke-tests the bundle (--selfcheck: profile + web assets load).
#
#   scripts/build_app.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || { echo "error: no venv at .venv — see CLAUDE.md dev setup" >&2; exit 1; }

"$PY" -m pip install -q -e ".[gui]" "pyinstaller>=6.10"
"$PY" -m PyInstaller --noconfirm --clean wobblemidi-gui.spec

APP=dist/wobblemidi.app
BIN="$APP/Contents/MacOS/wobblemidi"
[ -x "$BIN" ] || { echo "error: build produced no binary at $BIN" >&2; exit 1; }

echo "--- selfcheck (bundled profile + web assets) ---"
"$BIN" --selfcheck

VERSION=$("$PY" -c "import importlib.metadata as m; print(m.version('wobblemidi'))")
ZIP="dist/wobblemidi-${VERSION}-macos-$(uname -m).zip"
ditto -c -k --keepParent "$APP" "$ZIP"

echo "built: $APP"
echo "zip:   $ZIP"
