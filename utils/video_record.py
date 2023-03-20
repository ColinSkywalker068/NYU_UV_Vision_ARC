import errno
import cv2
import os
import numpy as np
from threading import Thread
import re
import config
from queue import Queue
import time

# definitions

SAVE_DIR = "/home/ultraviolet/recordings"
SAVE_NAME = "video"


class VideoRecord:
    def __init__(self):
        self.dim = config.camera['FRAME_SIZE']  # (640, 480)

        # Do not run if we are on windows
        if os.name == 'nt':
            print("Video recording is not supported on Windows")
            self.running = False
            return

        self.vid_idx = 0
        self.new_dir = ""
        self.running = True
        self.create_save_dir()
        self.create_video_dir()

        # 100 MB max size
        self.FREE_SPACE_BUFFER_KB = 1000000

        if self.running:
            self._init_writer()
            self.frame_queue = Queue()
            self.start_time = time.time()
            self.save_thread = Thread(target=self._write_thread)
            self.save_thread.start()

    # create_save_dir:
    # create save directory (SAVE_DIR) if it doesn't exist
    def create_save_dir(self):
        try:
            os.makedirs(SAVE_DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("Video save Directory already exists")
            else:
                print(e)
                self.running = False

    # create_video_dir: Create a new video subdirectory in main save directory
    # according to SAVE_DIR and SAVE_NAME, we increment the name of the
    # directory according to the previous max index
    def create_video_dir(self):
        vid_idx = [int(x.group(1)) for x in [re.search('recording_(\\d+)', x)
                                             for x in os.listdir(SAVE_DIR)] if x is not None]

        # in case vid_idx is empty
        vid_idx.append(0)

        self.new_dir = SAVE_DIR + "/recording_" + str(max(vid_idx) + 1)

        try:
            os.makedirs(self.new_dir)
            print("Created new video directory: " + self.new_dir)
        except OSError as e:
            self.running = False
            print(e)

    # _init_writer: Initialize the video writer according to the current
    # video index

    def _init_writer(self):
        self.vid_idx += 1

        codec = cv2.VideoWriter_fourcc(*'XVID')

        path = self.new_dir + "/" + str(self.vid_idx) + "_.avi"
        print("Creating new video file: " + path)

        self.writer = cv2.VideoWriter(path, codec, min(
            30, config.camera['FRAME_RATE']), self.dim, True)

    # _write_thread: Thread that writes frames to the video file
    # according to the current video index. If the video file is
    # larger than 100 MB or 30s has elapsed, we create a new video file and
    # increment the video index.
    def _write_thread(self):
        while self.running:
            os_info = os.statvfs(SAVE_DIR)
            free_space = os_info.f_bavail * os_info.f_frsize / 1024

            if free_space < self.FREE_SPACE_BUFFER_KB:
                self.running = False
                print("Device is out of space, stopping recording")
                break

            # nparray
            next_frame = self.frame_queue.get()
            if next_frame:
                self.writer.write(next_frame)

            c_time = time.time()

            # start a new video file if the current one is too large
            if c_time - self.start_time > 30:
                self.writer.release()
                self.start_time = c_time
                self._init_writer()
        self.writer.release()

    # process a new set of frames
    # called when getting a new camera frame
    def process_frame(self, frames):
        if not self.running:
            return

        if self.frame_queue.qsize() > 1:
            print("Video queue is full, dropping frame")
            self.frame_queue.get_nowait()

        self.frame_queue.put(frames)

    def __del__(self):
        if not self.running:
            return
        self.running = False
        self.frame_queue.put_nowait(None)
        self.save_thread.join(5)
