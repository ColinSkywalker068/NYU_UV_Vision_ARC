from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path("/home/ultraviolet/CV_Detection/legacy/models/bestmerge.pt")
DATA_YAML_PATH = Path("/home/ultraviolet/Datasets/mergeRM_final/data.yaml")
IMAGE_SIZE = 640
WORKSPACE_GB = 4


def clean_export_artifacts(model_path: Path) -> None:
    for ext in [".engine", ".onnx"]:
        model_path.with_suffix(ext).unlink(missing_ok=True)

    for search_dir in [Path.cwd(), model_path.parent]:
        for cache_path in search_dir.rglob("*calib*.cache"):
            cache_path.unlink(missing_ok=True)


def export_int8_engine() -> None:
    if not MODEL_PATH.is_file() or not DATA_YAML_PATH.is_file():
        raise FileNotFoundError("Missing model or data yaml")

    clean_export_artifacts(MODEL_PATH)

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
