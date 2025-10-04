import time

import os
from pathlib import Path

def acquire_file_lock(lock_path: Path, timeout_seconds: float = 60.0, poll_seconds: float = 0.1):
    """
    Acquire an inter-process file lock by atomically creating a lock file.
    Blocks (with polling) until acquired or timeout is reached.
    """
    start_time = time.time()
    lock_path = Path(lock_path)
    while True:
        try:
            # 'x' mode fails if file exists (atomic on same filesystem)
            with open(lock_path, 'x') as f:
                f.write(f"pid={os.getpid()} time={time.time()}\n")
            return
        except FileExistsError:
            if (time.time() - start_time) > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_seconds)

def release_file_lock(lock_path: Path):
    """Release the inter-process file lock by removing the lock file."""
    try:
        Path(lock_path).unlink(missing_ok=True)
    except Exception:
        # Best effort unlock
        pass


