from ultralytics import YOLO, RTDETR, NAS


class YOLODetector:
    def __init__(self, model_path, task="detect", model_type="yolo"):
        # 1) Store requested task.
        self.model = None
        self.task = task

        # 2) Build requested backend model.
        if model_type == "yolo":
            self.model = YOLO(model_path, task=task)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_path)
        elif model_type == "nas":
            self.model = NAS(model_path)
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")

    def track(self, image, **kwargs):
        """Wrapper for model.track()."""
        if "verbose" not in kwargs:
            # Default quiet mode keeps ROS logs usable at camera-frame rates.
            kwargs["verbose"] = False

        return self.model.track(image, **kwargs)

    def predict(self, image, **kwargs):
        """Wrapper for model.predict() or model()."""
        return self.model(image, **kwargs)
