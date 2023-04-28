import cv2
import time
import threading
from queue import Queue
from datetime import datetime
import os

class VideoRecord:
    def __init__(self, save_interval=10, file_prefix='output', file_extension='.mp4'):
        self.save_interval = save_interval
        self.file_prefix = file_prefix
        self.file_extension = file_extension
        self.frame_queue = Queue()
        self.frame_counter = 0
        self.saving_thread = threading.Thread(
            target=self.save_video_periodically)
        self.saving_thread.daemon = True
        self.saving_thread.start()

    def get_output_filename(self):
            video_dir = "~/Documents/video"
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
        print(f"Video saved as {filename}")
        print(f"Video duration: {len(frames) / 30.0:.2f} seconds")

    def save_video_periodically(self):
        frames_to_save = []
        while True:
            frame = self.frame_queue.get()
            frames_to_save.append(frame)
            self.frame_counter += 1

            if self.frame_counter >= 300:
                self.save_video(frames_to_save)
                frames_to_save = []
                self.frame_counter = 0

    def add_frame(self, frame):
        self.frame_queue.put(frame)


def main():
    cap = cv2.VideoCapture(0)
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
