"""Fast, non-mutating validation for the self-contained CoSkill image."""

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from pyserini.search.lucene import LuceneSearcher

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ALFWORLD_DATA = Path(os.environ.get("ALFWORLD_DATA", "/opt/data/alfworld"))
WEBSHOP_DATA = Path(os.environ.get(
    "WEBSHOP_DATA_DIR",
    PROJECT_ROOT / "agent_system/environments/env_package/webshop/webshop/data",
))
WEBSHOP_ROOT = WEBSHOP_DATA.parent
DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT", PROJECT_ROOT / "skillrl_data/verl-agent"))


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required asset: {path}")


for split in ("train", "valid_seen", "valid_unseen"):
    require(ALFWORLD_DATA / "json_2.1.1" / split)
require(ALFWORLD_DATA / "logic" / "alfred.pddl")
require(ALFWORLD_DATA / "logic" / "alfred.twl2")

for name in (
    "items_shuffle_1000.json",
    "items_ins_v2_1000.json",
    "items_human_ins.json",
):
    require(WEBSHOP_DATA / name)
require(WEBSHOP_ROOT / "search_engine" / "indexes" / "segments_3")
searcher = LuceneSearcher(str(WEBSHOP_ROOT / "search_engine" / "indexes"))
if searcher.num_docs != 1000:
    raise SystemExit(
        f"WebShop Lucene index has {searcher.num_docs} documents; expected 1000")

for name, expected in (("train.parquet", 12), ("test.parquet", 32)):
    path = DATA_ROOT / "text" / name
    require(path)
    rows = pq.ParquetFile(path).metadata.num_rows
    if rows != expected:
        raise SystemExit(f"{path} has {rows} rows; expected {expected}")

skills = PROJECT_ROOT / "memory_data"
for benchmark in ("alfworld", "webshop"):
    path = skills / benchmark / "claude_style_skills.json"
    require(path)
    with path.open(encoding="utf-8") as handle:
        json.load(handle)

output_root = Path(os.environ.get("OUTPUT_ROOT", "/outputs"))
output_root.mkdir(parents=True, exist_ok=True)
probe = output_root / ".coskill_write_probe"
probe.write_text("ok\n", encoding="utf-8")
probe.unlink()

print(f"ALFWorld data ready: {ALFWORLD_DATA}")
print(f"WebShop small data ready: {WEBSHOP_DATA}")
print("WebShop Lucene index ready: 1000 documents")
print(f"Prepared train/val parquet ready: {DATA_ROOT / 'text'}")
print(f"Output directory writable: {output_root}")
