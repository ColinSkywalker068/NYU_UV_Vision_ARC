from ultralytics import YOLO


def quantize_model():
    """Quantize YOLOv8 model to TensorRT INT8 engine."""
    model_path = "/workspace/CV_Detection/model_test/best.pt"
    yaml_path = "/workspace/RPS_Dataset/data.yaml"
    model = YOLO(model_path)
    model.export(data=yaml_path, format="engine", int8=True)
    print("done")


def main():
    quantize_model()
    return 0


main()
