#!/usr/bin/env python3
"""
YOLO26 OBB training entry point for RoboMaster armor plates.

This is tuned for the dataset analysis pasted into the thread:
  - train: 1600 images, 3124 OBB polygon labels
  - valid: 400 images, 747 OBB polygon labels
  - classes: red 1920 / blue 1204 in train, so use mild class weighting
  - size mix: ~20% tiny/small, ~35% medium, ~45% close/very-close
  - very-close examples are only ~5%, so use strong scale/multiscale
  - motion blur exists but is uneven, so add photometric + blur-only custom aug

Example:
  python3 legacy/training/train_robomaster_armor_obb.py \
    --dataset-root /path/to/dataset \
    --model yolo26s-obb.pt \
    --batch 16 \
    --device 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO26 OBB model for RoboMaster armor color detection."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset directory containing train/images and valid/images.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional existing data.yaml. If omitted, one is generated in dataset root.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s-obb.pt",
        help="Pretrained OBB model, e.g. yolo26n-obb.pt, yolo26s-obb.pt, yolo11s-obb.pt.",
    )
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="640 keeps the baseline speedier; raise it later if small targets still need help.",
    )
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="Use -1 for Ultralytics auto-batch, or set an integer for fixed batch.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/robomaster_obb")
    parser.add_argument("--name", type=str, default="armor_yolo26s_obb_close_motion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", default="disk", choices=["False", "ram", "disk"])
    parser.add_argument(
        "--export-engine",
        action="store_true",
        help="Export best.pt to TensorRT engine after training.",
    )
    parser.add_argument(
        "--no-custom-augmentations",
        action="store_true",
        help="Disable Albumentations-only blur/weather/lighting augmentation.",
    )
    return parser.parse_args()


def write_dataset_yaml(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    yaml_path = dataset_root / "robomaster_armor_obb.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: train/images",
                "val: valid/images",
                "# test: test/images",
                "",
                "names:",
                "  0: blue",
                "  1: red",
                "",
            ]
        )
    )
    return yaml_path


def validate_dataset(dataset_root: Path, data_yaml: Path) -> None:
    required = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected dataset paths:\n  " + "\n  ".join(missing))
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")


def build_custom_augmentations() -> list[Any]:
    """Photometric/blur-only transforms avoid changing OBB geometry."""
    try:
        import albumentations as A
    except ImportError:
        print("Albumentations is not installed; continuing with built-in YOLO aug only.")
        return []

    return [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 13), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.4), p=1.0),
            ],
            p=0.28,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.24,
            contrast_limit=0.22,
            p=0.35,
        ),
        A.HueSaturationValue(
            hue_shift_limit=6,
            sat_shift_limit=24,
            val_shift_limit=18,
            p=0.22,
        ),
        A.RandomGamma(gamma_limit=(75, 130), p=0.18),
        A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.08, 0.32), p=0.12),
        A.ImageCompression(quality_range=(55, 95), p=0.10),
    ]


def training_args(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    cache: bool | str = False if args.cache == "False" else args.cache
    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": int(args.batch) if args.batch >= 1 else args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "seed": args.seed,
        "deterministic": False,
        "cache": cache,
        "plots": True,
        "val": True,
        "save_period": 25,
        "optimizer": "auto",
        "cos_lr": True,
        "warmup_epochs": 5.0,
        "weight_decay": 0.0007,
        # Mild class weighting: train split is ~61% red / 39% blue.
        # Put extra pressure on rotated-box localization for small/fast armor.
        "box": 8.5,
        "angle": 1.25,
        "cls": 0.55,
        # Stronger scale/multiscale creates more extreme close-up and far-away cases.
        "multi_scale": 0.25,
        "degrees": 12.0,
        "translate": 0.18,
        "scale": 0.82,
        "shear": 3.0,
        "perspective": 0.00035,
        "fliplr": 0.5,
        # Upside-down full-frame flips are not common in field footage.
        "flipud": 0.02,
        # Keep mosaic high early for small/far targets, then turn it off near the end.
        "mosaic": 0.85,
        "close_mosaic": 35,
        # Low mix/cut helps occlusion robustness without washing out armor colors.
        "mixup": 0.06,
        "cutmix": 0.12,
        # Lighting/color robustness for LEDs, exposure shifts, and arena glare.
        "hsv_h": 0.018,
        "hsv_s": 0.62,
        "hsv_v": 0.42,
        "bgr": 0.02,
        "amp": True,
        "max_det": 50,
        "patience": 80,
        "verbose": True,
        "resume": args.resume,
    }
    if args.device is None:
        train_args.pop("device")
    if not args.no_custom_augmentations:
        custom_augs = build_custom_augmentations()
        if custom_augs:
            train_args["augmentations"] = custom_augs
    return train_args


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    data_yaml = args.data.resolve() if args.data else write_dataset_yaml(dataset_root)
    validate_dataset(dataset_root, data_yaml)

    from ultralytics import YOLO

    model = YOLO(args.model, task="obb")
    train_args = training_args(args, data_yaml)

    try:
        results = model.train(**train_args)
    except TypeError as exc:
        if "augmentations" not in train_args:
            raise
        print(f"Custom augmentations were rejected by this Ultralytics version: {exc}")
        print("Retrying with built-in YOLO augmentations only.")
        train_args.pop("augmentations", None)
        results = model.train(**train_args)

    best = Path(results.save_dir) / "weights" / "best.pt"
    if args.export_engine:
        if not best.exists():
            raise FileNotFoundError(f"Expected best checkpoint not found: {best}")
        export_model = YOLO(str(best), task="obb")
        export_model.export(
            format="engine",
            imgsz=args.imgsz,
            half=True,
            dynamic=True,
            simplify=True,
            data=str(data_yaml),
        )

    if best.exists():
        latest = Path(args.project) / "latest_best.pt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, latest)
        print(f"Best checkpoint: {best}")
        print(f"Copied shortcut: {latest}")


if __name__ == "__main__":
    main()
