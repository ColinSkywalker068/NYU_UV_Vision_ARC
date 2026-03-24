"""Export YOLO-OBB `best.pt` to INT8 TensorRT engine (Jetson AGX Orin) with calibration data.

Run on the Orin (Linux + TensorRT), not macOS. INT8 calibration uses images from the dataset
referenced by the YAML (default: mergeRM).

Example:
  cd /path/to/CV_Detection/model_test
  python3 quantize.py
  python3 quantize.py --data /home/ultraviolet/Datasets/mergeRM/data.yaml --workspace 8
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_DEFAULT_MERGE_RM_ROOT: Path = Path("/home/ultraviolet/Datasets/mergeRM")


def default_model_path() -> Path:
    return _SCRIPT_DIR / "best.pt"


def default_data_yaml() -> Path:
    env_path: str = os.environ.get("YOLO_CALIB_DATA", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    direct: Path = _DEFAULT_MERGE_RM_ROOT / "data.yaml"
    if direct.is_file():
        return direct
    alt: Path = _DEFAULT_MERGE_RM_ROOT / "dataset.yaml"
    if alt.is_file():
        return alt
    return direct


DEFAULT_MODEL_PATH: Path = default_model_path()
DEFAULT_IMAGE_SIZE: int = 640
# AGX Orin has ample unified memory; 8 GB workspace is safe for TensorRT builder.
DEFAULT_WORKSPACE_GB: int = int(os.environ.get("TRT_WORKSPACE_GB", "8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="INT8 TensorRT export for YOLO-OBB (Jetson AGX Orin; calibration via mergeRM)"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Weights (default: model_test/best.pt next to this script)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="YOLO data YAML for INT8 calibration (default: mergeRM data.yaml or YOLO_CALIB_DATA)",
    )
    parser.add_argument("--task", type=str, default="obb")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--workspace",
        type=int,
        default=DEFAULT_WORKSPACE_GB,
        help="TensorRT workspace size (GB)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("YOLO_EXPORT_DEVICE", "0"),
        help="CUDA device id for export (default: 0)",
    )
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


def resolve_data_path(args: argparse.Namespace) -> Optional[Path]:
    if args.data is not None:
        return args.data.expanduser().resolve()
    return default_data_yaml()


def validate_data_path(data_path: Optional[Path], export_target: str) -> None:
    if export_target != "engine":
        return
    if data_path is None or not data_path.is_file():
        raise FileNotFoundError(
            "TensorRT INT8 export needs a valid YOLO data YAML. "
            f"Expected under {_DEFAULT_MERGE_RM_ROOT} (e.g. data.yaml), or set --data / YOLO_CALIB_DATA. "
            f"Got: {data_path}"
        )


def export_model(args: argparse.Namespace) -> None:
    model_path: Path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"missing model: {model_path}")
    export_target: str = resolve_export_target(args.target)
    data_path: Optional[Path] = resolve_data_path(args)
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
            device=args.device,
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
