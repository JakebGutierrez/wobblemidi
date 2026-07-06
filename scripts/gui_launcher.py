"""PyInstaller entry stub for the GUI .app (see wobblemidi-gui.spec).

Imports wobblemidi_gui.app as a module — the same way the wobblemidi-gui
console script runs it — so its __file__-relative web-asset path resolves
inside the frozen package rather than at the bundle root.
"""

from wobblemidi_gui.app import main

if __name__ == "__main__":
    main()
