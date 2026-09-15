#!/usr/bin/env python3
"""Launch the dashboard:  ``streamlit run run.py``

Running this file directly (``python run.py``) re-execs it under streamlit, so
either invocation works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _under_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if _under_streamlit():
    from eis_dashboard.app import main
    main()
elif __name__ == "__main__":
    from eis_dashboard.settings import load
    port = str(load().port)
    os.execvp(sys.executable,
              [sys.executable, "-m", "streamlit", "run", __file__,
               "--server.port", port])
