from ultralytics import YOLO


def quantize_model():
    """Quantize YOLOv8 model to TensorRT INT8 engine."""
    model_path = "/home/ultraviolet/CV_Detection/yolov8_ros/models/best_new.pt"
    yaml_path = "/home/ultraviolet/Datasets/mergeRM/data.yaml"
    model = YOLO(model_path)
    model.export(data=yaml_path, format="engine", int8=True)
    print("done")


def main():
    quantize_model()
    return 0


main()
