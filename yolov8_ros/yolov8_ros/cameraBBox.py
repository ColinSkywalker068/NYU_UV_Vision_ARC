import cv2
import numpy as np
from ultralytics import YOLO


def check_enemy(roi_image, target_class, blue_ranges, red_ranges, color_threshold):
    if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
        print("Invalid ROI shape; returning False.")
        return False, None

    ranges = blue_ranges if target_class == "BLUE" else red_ranges
    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    mask = sum(
        cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1])
        for hsv_range in ranges
    )
    color_area = np.count_nonzero(mask)

    return color_area > color_threshold, mask


BLUE_RANGES = [
    [np.array((100, 150, 150), np.uint8), np.array((128, 255, 255), np.uint8)]
]
RED_RANGES = [
    [np.array((0, 150, 150), np.uint8), np.array((10, 255, 255), np.uint8)],
    [np.array((170, 150, 150), np.uint8), np.array((180, 255, 255), np.uint8)],
]

def main():
    # Load YOLO model
    # Adjust this path to your YOLO model
    model = YOLO("best.pt")  # You can use a different model like "best.pt" if you have it

    # Define target classes from YOLO model to analyze
    # 0: person, 39: bottle, etc. (COCO dataset classes)
    target_yolo_classes = [0, 39]  # Adjust based on what you want to detect

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    target_class = "RED"  # Default color detection target

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Run YOLO detection
        results = model(frame, stream=True)

        # Process results
        detected_objects = False

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Check if the detected class is in our target classes
                cls = int(box.cls[0].item())
                if cls in target_yolo_classes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Extract ROI
                    roi = frame[y1:y2, x1:x2]

                    # Skip empty ROIs
                    if roi.size == 0:
                        continue

                    # Calculate threshold based on ROI size
                    color_threshold = roi.shape[0] * roi.shape[1] * 0.01

                    # Check for enemy color in the ROI
                    result, mask = check_enemy(roi, target_class, BLUE_RANGES, RED_RANGES, color_threshold)

                    # Draw bounding box
                    color = (0, 255, 0) if result else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label
                    label = f"{model.names[cls]}: {target_class} {result}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detected_objects = True

                    # Display the ROI and its mask
                    if mask is not None:
                        # Resize mask to match ROI
                        mask_resized = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
                        mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                        roi_with_mask = np.hstack((roi, mask_bgr))

                        # Show ROI analysis in a separate window
                        cv2.imshow(f"ROI Analysis - {model.names[cls]}", roi_with_mask)

        # Show the main frame
        cv2.putText(
            frame,
            f"Detecting {target_class} in objects" if detected_objects else "No target objects detected",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.imshow("YOLO with Color Detection", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            target_class = "BLUE" if target_class == "RED" else "RED"
            print(f"Switched to {target_class} detection")
        elif key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
