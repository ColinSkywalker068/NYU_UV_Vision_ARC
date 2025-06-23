import cv2
import torch
import numpy as np
import time
import pyrealsense2 as rs
import os

def record_realsense_video(output_path, duration_minutes=5, width=640, height=480, fps=30):
    """
    Records a video from an Intel RealSense camera.

    Args:
        output_path (str): The path where the output video file will be saved.
        duration_minutes (int): The duration of the recording in minutes.
        width (int): The width of the video frame.
        height (int): The height of the video frame.
        fps (int): The framerate of the video.

    Returns:
        bool: True if recording was successful, False otherwise.
    """
    print("--- RealSense Recording ---")
    
    # --- 1. Initialize RealSense Pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure the color stream
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # --- 2. Initialize Video Writer ---
    # Define the codec and create VideoWriter object. 
    # 'XVID' is a common and reliable codec for .avi files on Linux.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for path {output_path}")
        return False
        
    print("Starting RealSense stream...")
    try:
        # Start the camera stream
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please ensure a RealSense camera is connected and the librealsense SDK is installed.")
        return False

    # --- 3. Recording Loop ---
    recording_duration_seconds = duration_minutes * 60
    start_time = time.time()
    
    print(f"Recording for {duration_minutes} minutes. Press 'q' in the video window to stop early.")
    
    try:
        while (time.time() - start_time) < recording_duration_seconds:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Write the frame to the output file
            out.write(color_image)
            
            # Display the resulting frame
            cv2.imshow('Recording... (press "q" to stop)', color_image)

            # Check for early exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped early by user.")
                break
    finally:
        # --- 4. Cleanup ---
        print("Stopping recording and cleaning up...")
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved successfully to: {output_path}")
        
    return True

def analyze_plate_detections(video_path, model_path, duration_minutes=5):
    """
    Analyzes a video file to calculate the average, low, and high range
    of red or blue plate detections per frame using a PyTorch model.

    Args:
        video_path (str): The path to the video file.
        model_path (str): The path to the bestObb2.pt model file.
        duration_minutes (int): The duration of the video to analyze in minutes.

    Returns:
        dict: A dictionary containing the analysis results, including
              average, min, and max detections, total frames processed,
              and total detections. Returns None if the video cannot be opened.
    """
    print("\n--- Plate Detection Analysis ---")
    # --- 1. Model Loading ---
    try:
        model = torch.load(model_path, map_location='cpu')['model'].float().eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # --- 2. Video Capture Initialization ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_to_process = int(duration_minutes * 60 * fps) if fps > 0 else 5 * 60 * 30 # Fallback
    
    print(f"Analyzing video with FPS: {fps}")
    print(f"Processing up to {total_frames_to_process} frames...")

    # --- 3. Detection Analysis ---
    detection_counts = []
    frames_processed = 0
    start_time = time.time()

    while cap.isOpened() and frames_processed < total_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # --- 4. Frame Pre-processing & Inference ---
        # Convert the frame from BGR to HSV color space as requested.
        # IMPORTANT: This assumes your model (`bestObb2.pt`) was trained on HSV images.
        # If it was trained on RGB images, this should be cv2.COLOR_BGR2RGB.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with torch.no_grad():
            results = model(img, size=640)

        # --- 5. Post-processing and Counting Detections ---
        current_frame_detections = 0
        if results.xyxy[0] is not None:
            predictions = results.xyxy[0]
            for *box, conf, cls in predictions:
                # Replace with your actual class indices for red/blue plates
                if int(cls) in [0, 1] and conf > 0.5:
                    current_frame_detections += 1
                        
        detection_counts.append(current_frame_detections)
        frames_processed += 1
        
        if frames_processed % 100 == 0:
            print(f"Processed {frames_processed} frames...")

    # --- 6. Cleanup and Analysis Calculation ---
    cap.release()
    end_time = time.time()
    
    print("\n--- Analysis Complete ---")
    print(f"Total time taken for analysis: {end_time - start_time:.2f} seconds")

    if not detection_counts:
        print("No frames were processed or no detections were made.")
        return {"average_detections_per_frame": 0, "min_detections_in_a_frame": 0, "max_detections_in_a_frame": 0}

    analysis_results = {
        "average_detections_per_frame": np.mean(detection_counts),
        "min_detections_in_a_frame": np.min(detection_counts),
        "max_detections_in_a_frame": np.max(detection_counts),
        "total_frames_with_detections": sum(1 for c in detection_counts if c > 0),
        "total_frames_processed": frames_processed,
        "total_detections": int(np.sum(detection_counts))
    }
    return analysis_results

# --- How to use the functions ---
if __name__ == '__main__':
    # --- Configuration ---
    RECORDING_DURATION_MIN = 5
    VIDEO_OUTPUT_FILE = 'realsense_capture.avi'
    MODEL_FILE = '/home/ultraviolet/CV_Detection/bestObb2.pt'

    # --- Step 1: Record Video ---
    # Make sure you have pyrealsense2 and a camera connected
    if record_realsense_video(VIDEO_OUTPUT_FILE, duration_minutes=RECORDING_DURATION_MIN):
        
        # --- Step 2: Analyze the Recorded Video ---
        # Check if the model file exists before analyzing
        if not os.path.exists(MODEL_FILE):
             print(f"Error: Model file not found at {MODEL_FILE}")
             print("Please update the MODEL_FILE variable with the correct path.")
        else:
            results = analyze_plate_detections(VIDEO_OUTPUT_FILE, MODEL_FILE, duration_minutes=RECORDING_DURATION_MIN)

            # --- Step 3: Print the Results ---
            if results:
                print("\n--- Detection Statistics ---")
                print(f"Frames Processed: {results['total_frames_processed']}")
                print(f"Total Detections (Red/Blue Plates): {results['total_detections']}")
                print("-" * 30)
                print(f"Average Detections per Frame: {results['average_detections_per_frame']:.2f}")
                print(f"Lowest Detection Count in a Single Frame: {results['min_detections_in_a_frame']}")
                print(f"Highest Detection Count in a Single Frame: {results['max_detections_in_a_frame']}")
                print("-" * 30)
