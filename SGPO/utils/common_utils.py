import json
import os
import tempfile
from pathlib import Path

import yaml


def load_json(path):
    print(f"Reading from {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.stem[:40]}.", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Saved to {path}")


def load_config(config_path):
    conf = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    print(f"Read config from {config_path}")
    return conf
