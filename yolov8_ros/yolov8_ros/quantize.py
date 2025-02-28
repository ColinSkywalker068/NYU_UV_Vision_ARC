from ultralytics import YOLO


def quantize_model():
    """Quantize YOLOv8 model to TensorRT INT8 engine."""
    model_path = "/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.pt/"
    model = YOLO(model_path)
    model.export(
        format="engine", int8=True, data="/home/ultraviolet/Downloads/trainTR/data.yaml"
    )
    print("done")


def main():
    quantize_model()
    return 0


main()
