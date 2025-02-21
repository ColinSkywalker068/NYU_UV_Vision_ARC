import sys
from ultralytics import YOLO


def quantize_model(model_path):
    """Quantize YOLOv8 model to TensorRT INT8 engine."""
    model = YOLO(model_path)
    model.export(format="engine", device=0, int8=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quantize_model.py <path_to_pt_model>")
        sys.exit(1)
    quantize_model(sys.argv[1])
