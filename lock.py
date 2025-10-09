import time
import sys
import os
from pathlib import Path
local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(add_date=False, config_path=Path(local_python_path)/ 'config.json')

def acquire_file_lock(lock_path: Path, timeout_seconds: float = 60.0, poll_seconds: float = 0.1):
    """
    Acquire an inter-process file lock by atomically creating a lock file.
    Blocks (with polling) until acquired or timeout is reached.
    """
    start_time = time.time()
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Acquiring lock: {lock_path}")
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
        logger.info(f"Released lock: {lock_path}")
    except Exception:
        # Best effort unlock
        pass


