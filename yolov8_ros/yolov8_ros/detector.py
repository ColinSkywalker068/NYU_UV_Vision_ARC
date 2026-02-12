from ultralytics import YOLO, RTDETR, NAS


class YOLODetector:
    def __init__(self, model_path, task="detect", model_type="yolo"):
        self.model = None
        self.task = task
        if model_type == "yolo":
            self.model = YOLO(model_path, task=task)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_path)
        elif model_type == "nas":
            self.model = NAS(model_path)
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")
        self._normalize_class_names()

    def _normalize_class_names(self):
        """
        Ensure class-name keys are integer ids.
        Some exported engines can load names as string keys (e.g., "0", "1"),
        which later causes KeyError when Ultralytics indexes with ints.
        """
        names = getattr(self.model, "names", None)
        if not isinstance(names, dict):
            return
        normalized = {}
        for key, value in names.items():
            normalized_key = key
            if isinstance(key, str) and key.isdigit():
                normalized_key = int(key)
            normalized[normalized_key] = value
        self.model.names = normalized

    def track(self, image, **kwargs):
        """
        Wrapper for model.track()
        """
        if "verbose" not in kwargs:
            kwargs["verbose"] = False
        return self.model.track(image, **kwargs)

    def predict(self, image, **kwargs):
        """
        Wrapper for model.predict() or model()
        """
        return self.model(image, **kwargs)
