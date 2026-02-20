from ultralytics import YOLO


def quantize_model():
    """Quantize YOLOv8 model to TensorRT INT8 engine."""
    model_path = "/home/ultraviolet/CV_Detection/yolov8_ros/models/best_new.pt"
    yaml_path = "/home/ultraviolet/Datasets/mergeRM/data.yaml"
    model = YOLO(model_path)
    # Critical: Specify imgsz to match training size (640x640 square)
    # Training uses imgsz=640, so quantization must match
    # Inference can use 640x480 with letterboxing, but model expects square input
    #
    # For "missing scale and zero-point": use dynamic=False (fixed shape lets TRT
    # calibrate all layers), batch=1 for calibration. If error persists, delete
    # any existing *_int8_calib.cache next to the engine.
    model.export(
        data=yaml_path,
        format="engine",
        int8=True,
        imgsz=640,  # Square input size matching training (imgsz=640)
        half=False,  # INT8 quantization, don't use FP16
        simplify=True,  # Simplify ONNX before TensorRT conversion
        workspace=4,  # TensorRT workspace size in GB
        dynamic=False,  # Fixed input shape so TRT can compute scale/zero-point for all tensors
        batch=1,  # Calibration batch size; val set must have >= this many images
    )
    print("Quantization complete. Model exported with imgsz=640 (matches training)")


def main():
    quantize_model()
    return 0


main()
