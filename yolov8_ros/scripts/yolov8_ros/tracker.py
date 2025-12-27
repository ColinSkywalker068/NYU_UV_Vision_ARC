import cv2
import numpy as np
import random


class ArmorPlateTracker:
    def __init__(self, initial_x, initial_y, width, height):
        self.kf = cv2.KalmanFilter(
            6, 2
        )  # 6 states (x,y,vx,vy,ax,ay), 2 measurements (x,y)

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32
        )

        # --- TUNABLE KALMAN PARAMETERS ---
        proc_noise_pos = 0.01
        proc_noise_vel = 0.05
        proc_noise_acc = 0.1

        self.kf.processNoiseCov = (
            np.diag(
                [
                    proc_noise_pos,
                    proc_noise_pos,
                    proc_noise_vel,
                    proc_noise_vel,
                    proc_noise_acc,
                    proc_noise_acc,
                ]
            ).astype(np.float32)
            * 1.0
        )

        meas_noise = 0.15
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise

        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1

        # Initial state: [x, y, vx, vy, ax, ay]'
        self.kf.statePost = np.array(
            [np.float32(initial_x), np.float32(initial_y), 0, 0, 0, 0], np.float32
        ).reshape(6, 1)

        self.width = float(width)
        self.height = float(height)

    def _update_transition_matrix(self, dt):
        # State transition matrix for constant acceleration model
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt * dt, 0],
                [0, 1, 0, dt, 0, 0.5 * dt * dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

    def predict(self, dt):
        self._update_transition_matrix(dt)
        prediction_state = self.kf.predict()
        return int(prediction_state[0, 0]), int(prediction_state[1, 0])  # x, y

    def update(self, x, y, width, height):
        measurement = np.array([[np.float32(x)], [np.float32(y)]], np.float32)
        self.kf.correct(measurement)
        self.width = float(width)
        self.height = float(height)


class TrackerWrapper:
    def __init__(self, tracker: ArmorPlateTracker):
        self.tracker = tracker
        self.lost_frames = 0
