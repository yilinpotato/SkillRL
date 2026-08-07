#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def check_alfworld() -> None:
    root = Path(os.environ.get("ALFWORLD_DATA", ROOT / "data" / "alfworld"))
    require(root / "logic" / "alfred.pddl", "ALFWorld PDDL")
    require(root / "logic" / "alfred.twl2", "ALFWorld TextWorld logic")
    for split in ("train", "valid_seen", "valid_unseen"):
        require(root / "json_2.1.1" / split, f"ALFWorld {split} split")
    print(f"ALFWorld: {root}")


def check_webshop() -> None:
    data = Path(
        os.environ.get(
            "WEBSHOP_DATA_DIR",
            ROOT / "agent_system/environments/env_package/webshop/webshop/data",
        )
    )
    for name in (
        "items_shuffle_1000.json",
        "items_ins_v2_1000.json",
        "items_human_ins.json",
    ):
        require(data / name, f"WebShop {name}")
    index = data.parent / "search_engine" / "indexes"
    if not any(index.glob("segments_*")):
        raise SystemExit(f"missing WebShop Lucene index: {index}")
    print(f"WebShop: {data}")


def check_common(allow_missing_parquet: bool) -> None:
    model = Path(
        os.environ.get(
            "MODEL_PATH",
            ROOT
            / ".cache/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507",
        )
    )
    require(model / "config.json", "model config")

    for benchmark in ("alfworld", "webshop"):
        path = ROOT / "memory_data" / benchmark / "initial_skills.json"
        require(path, f"{benchmark} skill library")
        with path.open(encoding="utf-8") as handle:
            json.load(handle)

    data_root = Path(os.environ.get("DATA_ROOT", ROOT / "data" / "verl-agent"))
    parquet_paths = []
    for name in ("train.parquet", "test.parquet"):
        path = data_root / "text" / name
        if path.exists() or not allow_missing_parquet:
            require(path, name)
        if path.exists():
            parquet_paths.append(path)
    if len(parquet_paths) == 2:
        import pyarrow.parquet as pq

        expected_rows = {"train.parquet": 12, "test.parquet": 32}
        for path in parquet_paths:
            rows = pq.ParquetFile(path).metadata.num_rows
            if rows != expected_rows[path.name]:
                raise SystemExit(
                    f"{path} has {rows} rows; expected {expected_rows[path.name]}"
                )
    print(f"Model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("alfworld", "webshop", "all"),
        default="all",
    )
    parser.add_argument("--allow-missing-parquet", action="store_true")
    args = parser.parse_args()

    check_common(args.allow_missing_parquet)
    if args.benchmark in ("alfworld", "all"):
        check_alfworld()
    if args.benchmark in ("webshop", "all"):
        check_webshop()


if __name__ == "__main__":
    main()
