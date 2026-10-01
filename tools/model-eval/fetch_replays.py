#!/usr/bin/env python3
"""Fetch pinned public RoboMaster test videos with source/hash provenance.

Keep clips and derived overlays outside git. Public demo footage is unlabeled and
may overlap model training data; never use these runs as held-out accuracy scores.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path

import fetch_candidates as fetcher

RP = "SZURPVision/26_NNDeployment_Lib_and_Detection_Models"
REV = "ccb5dfe09696edeea445ffb8f99adaa7c383ec3a"
ITEMS = [
    fetcher.github_item("TongjiSuperPower/sp_vision_25", "bd9f5e798fa3c6dd3b483ae6627796afb41c608d",
                        "assets/demo/demo.avi", "tongji-demo.avi", 62652416,
                        "95f1dbc56545a6d972da082d1c186363c4dc90e2"),
    fetcher.github_item(RP, REV, "测试视频/装甲板.mp4", "robotpilots-armor.mp4", 2248213,
                        "656017d6eeb6a5879a36ccca7473e03638212323"),
    fetcher.github_item(RP, REV, "测试视频/中距离陀螺.avi", "robotpilots-medium-spin.avi", 90813600,
                        "1de611e959dd2cc52cce5af3a07cd0c02b1e540d"),
    fetcher.github_item(RP, REV, "测试视频/近距离上下陀螺.avi", "robotpilots-near-height-spin.avi", 74767390,
                        "cd66b5e97f8028861acc71909eb3acf4f7f5c5d9"),
    fetcher.github_item(RP, REV, "测试视频/远距离平移.avi", "robotpilots-far-translation.avi", 53402968,
                        "074c54bb948cf9266a544b22a528f92461fa92d0"),
    fetcher.github_item(RP, REV, "测试视频/远处旋转平移.avi", "robotpilots-far-translation-spin.avi", 64943384,
                        "179877635e773d596e35550a25849d3b3c3989cb"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", action="append", choices=[i["name"] for i in ITEMS])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fetcher.DEST = args.output.resolve()
    selected = [i for i in ITEMS if not args.clip or i["name"] in args.clip]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetcher.fetch, selected))
    (args.output / "replay-manifest.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    for item in results:
        print(json.dumps({k: item[k] for k in ("name", "size", "verified", "sha256")}))


if __name__ == "__main__":
    main()
