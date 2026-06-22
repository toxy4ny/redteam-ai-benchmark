"""Append-only JSONL request diagnostics."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def append_request_log(path: Optional[str], entry: Dict[str, Any]) -> None:
    """Append one request diagnostics record when logging is configured."""
    if not path:
        return

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **entry,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
