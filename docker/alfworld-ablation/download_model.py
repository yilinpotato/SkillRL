"""Download the configured ModelScope snapshot into a stable container path."""
import os
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


model_id = os.environ.get("MODELSCOPE_MODEL_ID", "Qwen/Qwen3-4B-Thinking-2507")
model_path = Path(os.environ["MODEL_PATH"])
model_path.mkdir(parents=True, exist_ok=True)
snapshot_download(model_id=model_id, local_dir=str(model_path))
if not (model_path / "config.json").is_file():
    raise RuntimeError(f"ModelScope download completed but config.json is missing under {model_path}")
print(f"Model ready: {model_id} -> {model_path}")
