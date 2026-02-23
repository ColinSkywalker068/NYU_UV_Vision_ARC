from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path("/home/ultraviolet/CV_Detection/yolov8_ros/models/best_new.pt")
DATA_YAML_PATH = Path("/home/ultraviolet/Datasets/mergeRM/data.yaml")
IMAGE_SIZE = 640
WORKSPACE_GB = 4


def remove_calibration_cache(model_path: Path) -> None:
    for cache_path in model_path.parent.glob("*calib*.cache"):
        cache_path.unlink(missing_ok=True)
        print(f"removed calibration cache: {cache_path}")


def export_int8_engine() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"missing model: {MODEL_PATH}")
    if not DATA_YAML_PATH.is_file():
        raise FileNotFoundError(f"missing data yaml: {DATA_YAML_PATH}")
    remove_calibration_cache(MODEL_PATH)
    model = YOLO(str(MODEL_PATH), task="obb")
    model.export(
        format="engine",
        int8=True,
        half=False,
        data=str(DATA_YAML_PATH),
        imgsz=IMAGE_SIZE,
        dynamic=False,
        batch=1,
        workspace=WORKSPACE_GB,
        simplify=True,
    )
    print("INT8 export done")


def main() -> int:
    export_int8_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
