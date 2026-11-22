#!/usr/bin/env python3
"""
ShieldEdge Root CLI Runner
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(BASE_DIR, "shieldedge")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from ui_cli import main

if __name__ == "__main__":
    main()
