import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def check_enemy(roi_image, target_class, blue_ranges, red_ranges, color_threshold):
    if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
        print("Invalid ROI shape; returning False.")
        return False, None

    ranges = blue_ranges if target_class == "BLUE" else red_ranges
    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    mask = sum(cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1]) for hsv_range in ranges)
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
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Load YOLO model
    model = YOLO("best.pt")

    # Define target classes from YOLO model
    target_yolo_classes = [0, 39]  # Adjust based on what you want to detect

    target_class = "RED"  # Default color detection target

    try:
        while True:
            # Wait for a frameset
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())

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
                        result, mask = check_enemy(
                            roi, target_class, BLUE_RANGES, RED_RANGES, color_threshold
                        )
                        if result:
                            # Display pause notification
                            cv2.putText(
                                frame,
                                "TARGET DETECTED - PAUSING FOR 2 SECONDS",
                                (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )

                            # Create combined view with mask for display during pause
                            if mask is not None:
                                mask_resized = cv2.resize(
                                    mask, (roi.shape[1], roi.shape[0])
                                )
                                mask_bgr = cv2.cvtColor(
                                    mask_resized, cv2.COLOR_GRAY2BGR
                                )
                                roi_with_mask = np.hstack((roi, mask_bgr))

                                # Add overlay to main display
                                h, w = roi_with_mask.shape[:2]
                                overlay_y, overlay_x = 70, 20
                                frame[
                                    overlay_y : overlay_y + h, overlay_x : overlay_x + w
                                ] = roi_with_mask

                            # Show frame with pause notification
                            cv2.imshow("YOLO with Color Detection", frame)
                            cv2.waitKey(2000)  # Pause for 2 seconds
                        # Draw bounding box
                        color = (0, 255, 0) if result else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Label
                        label = f"{model.names[cls]}: {target_class} {result}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                        detected_objects = True

                        # Display the ROI and its mask
                        if mask is not None:
                            # Display the ROI and its mask
                            if mask is not None:
                                # Resize mask to match ROI
                                mask_resized = cv2.resize(
                                    mask, (roi.shape[1], roi.shape[0])
                                )
                                mask_bgr = cv2.cvtColor(
                                    mask_resized, cv2.COLOR_GRAY2BGR
                                )
                                roi_with_mask = np.hstack((roi, mask_bgr))

                                # Add color_area value to the display
                                color_area = np.count_nonzero(mask)
                                color_percentage = (
                                    color_area / (roi.shape[0] * roi.shape[1])
                                ) * 100
                                color_area_text = f"Color area: {color_area} ({color_percentage:.1f}%)"
                                cv2.putText(
                                    roi_with_mask,
                                    color_area_text,
                                    (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                )

                                # Show ROI analysis in a separate window
                                cv2.imshow(
                                    f"ROI Analysis - {model.names[cls]}", roi_with_mask
                                )
            # Show the main frame
            cv2.putText(
                frame,
                f"Detecting {target_class} in objects"
                if detected_objects
                else "No target objects detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("YOLO with Color Detection", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("t"):
                target_class = "BLUE" if target_class == "RED" else "RED"
                print(f"Switched to {target_class} detection")
            elif key == ord("q"):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
