import argparse
import platform
from pathlib import Path
from typing import Optional

import yaml

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_PATH = LEGACY_ROOT / "models" / "best.pt"
DEFAULT_DATA_PATH = SCRIPT_DIR / "mergeRM/data.yaml"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_WORKSPACE_GB = 4
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model for deployment")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--task", type=str, default="obb")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--workspace", type=int, default=DEFAULT_WORKSPACE_GB)
    parser.add_argument(
        "--target",
        type=str,
        choices=("auto", "engine", "coreml"),
        default="auto",
    )
    return parser.parse_args()


def resolve_export_target(target: str) -> str:
    if target != "auto":
        return target
    if platform.system() == "Darwin":
        return "coreml"
    return "engine"


def remove_calibration_cache(model_path: Path) -> None:
    for cache_path in model_path.parent.glob("*calib*.cache"):
        cache_path.unlink(missing_ok=True)
        print(f"removed calibration cache: {cache_path}")


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_data_path(data_path: Optional[Path]) -> Path:
    if data_path is None:
        raise FileNotFoundError("missing quantization data yaml")
    resolved_data_path = resolve_input_path(data_path.expanduser())
    if not resolved_data_path.is_file():
        raise FileNotFoundError(f"missing data yaml: {resolved_data_path}")
    return resolved_data_path


def count_images(directory_path: Path) -> int:
    if not directory_path.is_dir():
        return 0
    return sum(
        1
        for file_path in directory_path.iterdir()
        if file_path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_local_data_yaml(data_path: Path) -> Path:
    data_config = yaml.safe_load(data_path.read_text())
    dataset_root = data_path.parent.resolve()
    data_config["path"] = str(dataset_root)
    generated_dir = SCRIPT_DIR / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    local_data_path = generated_dir / f"{data_path.stem}.local.yaml"
    local_data_path.write_text(yaml.safe_dump(data_config, sort_keys=False))
    return local_data_path


def validate_merge_rm_dataset(data_path: Path, task: str) -> Path:
    local_data_path = build_local_data_yaml(data_path)
    dataset_root = data_path.parent.resolve()
    val_images_dir = dataset_root / "images" / "val"
    train_images_dir = dataset_root / "images" / "train"
    test_images_dir = dataset_root / "images" / "test"
    val_image_count = count_images(val_images_dir)
    train_image_count = count_images(train_images_dir)
    test_image_count = count_images(test_images_dir)
    if val_image_count == 0:
        raise FileNotFoundError(
            "mergeRM is missing validation images at "
            f"{val_images_dir}. INT8 export needs real images for calibration."
        )
    if train_image_count == 0 and test_image_count == 0:
        print(
            "warning: mergeRM only has validation images locally. "
            "Calibration can still run, but the dataset copy looks incomplete."
        )
    if task != "obb":
        print(
            "warning: mergeRM labels are OBB polygons. "
            "Using a non-obb task may fail or load the dataset incorrectly."
        )
    return local_data_path


def export_model(args: argparse.Namespace) -> None:
    model_path: Path = resolve_input_path(args.model.expanduser())
    if not model_path.is_file():
        raise FileNotFoundError(f"missing model: {model_path}")
    export_target = resolve_export_target(args.target)
    data_path = resolve_data_path(args.data)
    local_data_path = validate_merge_rm_dataset(data_path, args.task)
    print(f"using quantization data: {local_data_path}")
    model = YOLO(str(model_path), task=args.task)
    if export_target == "engine":
        remove_calibration_cache(model_path)
        model.export(
            format="engine",
            int8=True,
            half=False,
            data=str(local_data_path),
            imgsz=args.imgsz,
            dynamic=False,
            batch=1,
            workspace=args.workspace,
            simplify=True,
        )
        print(f"exported TensorRT INT8 engine next to {model_path.name}")
        return
    export_kwargs = {
        "format": "coreml",
        "int8": True,
        "data": str(local_data_path),
        "imgsz": args.imgsz,
    }
    if args.task == "detect":
        export_kwargs["nms"] = True
    model.export(**export_kwargs)
    print(f"exported CoreML INT8 package next to {model_path.name}")


def main() -> int:
    args = parse_args()
    export_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
