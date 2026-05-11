import os
from pathlib import Path

def get_local_path(key: str) -> Path:
    # "data_lake/..."
    local_path = Path("data_lake") / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return local_path
