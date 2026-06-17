import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class CaptureBuffer:
    """In-memory buffer flushed to parquet files (one per flush) when full.

    Rows are dicts mixing input features and the predicted value. Files land in
    `root/YYYY-MM-DD/HH-<uuid>.parquet` so they can be globbed by date or hour.
    """

    def __init__(self, root: str | Path, flush_size: int = 50) -> None:
        self.root = Path(root)
        self.flush_size = flush_size
        self._lock = threading.Lock()
        self._buffer: list[dict] = []

    def append(self, row: dict) -> None:
        row = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            **row,
        }
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self.flush_size:
                self._flush_locked()

    def flush(self) -> int:
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> int:
        if not self._buffer:
            return 0
        rows = self._buffer
        self._buffer = []
        now = datetime.now(timezone.utc)
        day_dir = self.root / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{now.strftime('%H')}-{uuid.uuid4().hex[:8]}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return len(rows)


def buffer_from_env() -> CaptureBuffer:
    root = os.environ.get("CAPTURE_DIR", "/tmp/predictions_log")
    flush_size = int(os.environ.get("CAPTURE_FLUSH_SIZE", "50"))
    return CaptureBuffer(root=root, flush_size=flush_size)
