import argparse
import platform
from pathlib import Path
from typing import Iterator, Union

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

DEFAULT_ENGINE_PATH = Path("yolov8_ros/models/best.engine")
DEFAULT_PT_PATH = Path("yolov8_ros/models/best_jithendra.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal YOLO + ByteTrack runner")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--task", type=str, default="obb")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def resolve_model_path(model_path: Union[Path, None]) -> Path:
    if model_path is not None:
        resolved_model_path = model_path.expanduser().resolve()
        if not (resolved_model_path.is_file() or resolved_model_path.is_dir()):
            raise FileNotFoundError(f"missing model: {resolved_model_path}")
        return resolved_model_path
    default_paths = (DEFAULT_ENGINE_PATH, DEFAULT_PT_PATH)
    for candidate_path in default_paths:
        resolved_candidate_path = candidate_path.expanduser().resolve()
        if resolved_candidate_path.is_file():
            return resolved_candidate_path
    raise FileNotFoundError(
        f"missing default model. checked {DEFAULT_ENGINE_PATH} and {DEFAULT_PT_PATH}"
    )


def resolve_source(source: str) -> Union[int, str]:
    if source.isdigit():
        return int(source)
    return source


def resolve_device(device: Union[str, None], model_path: Path) -> Union[int, str, None]:
    if device is not None:
        return int(device) if device.isdigit() else device
    if model_path.suffix == ".engine":
        return 0
    if platform.system() == "Darwin":
        return "mps"
    return None


def stream_tracks(
    model: YOLO,
    source: Union[int, str],
    device: Union[int, str, None],
    conf: float,
    imgsz: int,
    save: bool,
) -> Iterator[Results]:
    return model.track(
        source=source,
        conf=conf,
        imgsz=imgsz,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
        show=False,
        save=save,
        device=device,
    )


def run_tracking(args: argparse.Namespace) -> None:
    model_path = resolve_model_path(args.model)
    source = resolve_source(args.source)
    device = resolve_device(args.device, model_path)
    model = YOLO(str(model_path), task=args.task)
    for result in stream_tracks(
        model=model,
        source=source,
        device=device,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
    ):
        annotated_frame = result.plot()
        cv2.imshow("YOLO ByteTrack", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    run_tracking(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
