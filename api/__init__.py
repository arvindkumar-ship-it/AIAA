"""
Package marker for `api`.

This file only exists so that the flat import used inside routes.py
(`from agent_orchestrator import ...`) can resolve when the app is
started from the project root via `python main.py`. It does not
touch any logic from the original modules.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
