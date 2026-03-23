import argparse
import platform
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

DEFAULT_MODEL_PATH = Path("yolov8_ros/models/best_jithendra_tuned.pt")
DEFAULT_IMAGE_SIZE = 640
DEFAULT_WORKSPACE_GB = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model for deployment")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=None)
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


def validate_data_path(data_path: Optional[Path], export_target: str) -> None:
    if export_target != "engine":
        return
    if data_path is None or not data_path.is_file():
        raise FileNotFoundError(
            "TensorRT INT8 export needs --data pointing to a valid data.yaml file."
        )


def export_model(args: argparse.Namespace) -> None:
    model_path: Path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"missing model: {model_path}")
    export_target = resolve_export_target(args.target)
    data_path: Optional[Path] = None
    if args.data is not None:
        data_path = args.data.expanduser().resolve()
    validate_data_path(data_path, export_target)
    model = YOLO(str(model_path), task=args.task)
    if export_target == "engine":
        remove_calibration_cache(model_path)
        model.export(
            format="engine",
            int8=True,
            half=False,
            data=str(data_path),
            imgsz=args.imgsz,
            dynamic=False,
            batch=1,
            workspace=args.workspace,
            simplify=True,
        )
        print(f"exported TensorRT INT8 engine next to {model_path.name}")
        return
    model.export(
        format="coreml",
        int8=True,
        imgsz=args.imgsz,
        nms=True,
    )
    print(f"exported CoreML INT8 package next to {model_path.name}")


def main() -> int:
    args = parse_args()
    export_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
