# python -m dox_pipeline.file_utils

import hashlib
import time

def generate_file_id(file_path: str, retries: int = 5, delay: int = 1) -> str:
    for attempt in range(retries):
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except PermissionError:
            time.sleep(delay)
    raise Exception(f"Failed to read {file_path} after {retries} attempts.")
