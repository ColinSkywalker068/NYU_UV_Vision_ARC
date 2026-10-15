#!/usr/bin/env python3
"""Prepend Ultralytics-compatible JSON metadata to a TensorRT plan."""

import argparse
import json
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw TensorRT plan")
    parser.add_argument("output", type=Path, help="Metadata-wrapped engine")
    parser.add_argument("--version", default="unknown", help="Exporter version")
    parser.add_argument("--class-name", action="append", required=True,
                        help="Class name in index order; repeat for each class")
    parser.add_argument("--task", default="obb", help="Model task")
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--int8", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = {
        "description": "Device-local TensorRT engine",
        "version": args.version,
        "task": args.task,
        "names": {str(index): name for index, name in enumerate(args.class_name)},
        "end2end": args.end2end,
        "int8": args.int8,
        "nms": args.nms,
    }
    encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    plan = args.input.read_bytes()
    args.output.write_bytes(struct.pack("<I", len(encoded)) + encoded + plan)
    print(f"Wrote {args.output} ({len(encoded)} metadata bytes, {len(plan)} plan bytes)")


if __name__ == "__main__":
    main()
