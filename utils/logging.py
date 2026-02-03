"""
utils/logging.py

Simple logging utilities for the demo.
- Works in notebooks, CLI, or Streamlit.
- Keeps logs readable and suitable for a demo narrative.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _ts() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def log_info(msg: str, **kv: Any) -> str:
    """Return a formatted log line (caller can print or store)."""
    suffix = ""
    if kv:
        suffix = " | " + " ".join([f"{k}={v}" for k, v in kv.items()])
    return f"[{_ts()}] INFO  {msg}{suffix}"


def log_warn(msg: str, **kv: Any) -> str:
    suffix = ""
    if kv:
        suffix = " | " + " ".join([f"{k}={v}" for k, v in kv.items()])
    return f"[{_ts()}] WARN  {msg}{suffix}"


def log_error(msg: str, **kv: Any) -> str:
    suffix = ""
    if kv:
        suffix = " | " + " ".join([f"{k}={v}" for k, v in kv.items()])
    return f"[{_ts()}] ERROR {msg}{suffix}"
