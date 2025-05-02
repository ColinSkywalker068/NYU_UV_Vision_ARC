import onnxruntime as ort
import numpy as np

# Load your ONNX model
session = ort.InferenceSession("/home/ultraviolet/CV_Detection/yolov8_ros/models/HUST_model.onnx")

# Prepare your input (replace with your actual input shape and data)
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})

print(outputs)