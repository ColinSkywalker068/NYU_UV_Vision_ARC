from ultralytics import YOLO


def quantize_model():
    """Quantize YOLOv8 model to TensorRT INT8 engine."""
    model_path = "/home/ultraviolet/CV_Detection/model_test/best.pt/"
    model = YOLO(model_path)
    model.export(
        format="engine", int8=True
    )
    print("done")


def main():
    quantize_model()
    return 0


main()
