from ultralytics import YOLO


def quantize_model():
    """Quantize YOLOv8 model to TensorRT INT8 engine."""
    model_path = "/home/ultraviolet/CV_Detection/yolov8_ros/models/best_new.pt"
    yaml_path = "/home/ultraviolet/Datasets/mergeRM/data.yaml"
    model = YOLO(model_path)
    # Critical: Specify imgsz to match training size (640x640 square)
    # Training uses imgsz=640, so quantization must match
    # Inference can use 640x480 with letterboxing, but model expects square input
    model.export(
        data=yaml_path,
        format="engine",
        int8=True,
        imgsz=640,  # Square input size matching training (imgsz=640)
        half=False,  # INT8 quantization, don't use FP16
        simplify=True,  # Simplify ONNX before TensorRT conversion
        workspace=4,  # TensorRT workspace size in GB
    )
    print("Quantization complete. Model exported with imgsz=640 (matches training)")


def main():
    quantize_model()
    return 0


main()
