# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the wobblemidi GUI macOS .app.
# Build with scripts/build_app.sh — do not run pyinstaller by hand unless you
# know why. Checked in (gitignore has an explicit exception) so the build is
# scripted and repeatable.

a = Analysis(
    # Entry stub, NOT wobblemidi_gui/app.py directly: the entry script runs at
    # the bundle root, which would break app.py's __file__-relative web path.
    ["scripts/gui_launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[
        # Bundled profile, resolved at runtime via importlib.resources —
        # must live at <bundle>/wobblemidi/profiles/ to match the package path.
        ("wobblemidi/profiles/rock.json", "wobblemidi/profiles"),
        # Front-end assets, resolved via Path(__file__).parent / "web".
        ("wobblemidi_gui/web", "wobblemidi_gui/web"),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["pandas", "pytest"],  # dev/scripts-only deps; keep the bundle smaller
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="wobblemidi",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="wobblemidi",
)

app = BUNDLE(
    coll,
    name="wobblemidi.app",
    icon=None,
    bundle_identifier="com.jakebgutierrez.wobblemidi",
    version="1.0.0",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
