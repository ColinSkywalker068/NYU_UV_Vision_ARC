import cv2
import time
import threading
from queue import Queue
from datetime import datetime
import os
import config
from pathlib import Path
import logging
logging.basicConfig(
    filename=config.general["LOG_PATH"],
    filemode='a',
    level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)

class VideoRecord:
    def __init__(self, save_interval=5, file_prefix='output', file_extension='.mp4'):
        self.save_interval = save_interval
        self.file_prefix = file_prefix
        self.file_extension = file_extension
        self.frame_queue = Queue()
        self.frame_counter = 0
        self.saving_thread = threading.Thread(target=self.save_video_periodically)
        self.saving_thread.daemon = True
        self.saving_thread.start()

    def get_output_filename(self):
            video_dir ="./video"
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return f"{video_dir}/{self.file_prefix}_{current_time}{self.file_extension}"

    def save_video(self, frames):
        if len(frames) == 0:
            return
        filename = self.get_output_filename()
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        logger.debug(f"Video saved as {filename}")
        logger.debug(f"Video duration: {len(frames) / 30.0:.2f} seconds")

    def save_video_periodically(self):
        frames_to_save = []
        while True:
            frame = self.frame_queue.get()
            frames_to_save.append(frame)
            self.frame_counter += 1

            # Change the frame threshold to 150 (30 FPS * 5 seconds = 150 frames)
            if self.frame_counter >= 600:
                self.save_video(frames_to_save)
                frames_to_save = []
                self.frame_counter = 0

    def add_frame(self, frame):
        self.frame_queue.put(frame)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Preheat the camera by reading a few frames before recording
    preheat_seconds = 1
    preheat_frames = preheat_seconds * 30  # 1 seconds * 30 FPS = 150 frames
    for _ in range(preheat_frames):
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)

    video_saver = VideoRecord()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_saver.add_frame(frame)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
