import numpy as np


class ArmorPlateTracker:
    """6-state (x, y, vx, vy, ax, ay) linear Kalman filter using NumPy."""

    def __init__(
        self,
        x,
        y,
        width,
        height,
        proc_noise=(0.01, 0.05, 0.1),
        meas_noise=0.15,
        init_cov=0.1,
    ):
        # State: [x, y, vx, vy, ax, ay]
        self.state = np.array([x, y, 0, 0, 0, 0], dtype=np.float64)

        pn_pos, pn_vel, pn_acc = proc_noise
        self.Q = np.diag([pn_pos, pn_pos, pn_vel, pn_vel, pn_acc, pn_acc])
        self.R = np.eye(2) * meas_noise
        self.P = np.eye(6) * init_cov

        # Measurement matrix: observe x and y only
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        self.width = float(width)
        self.height = float(height)

    def _transition(self, dt):
        dt2 = 0.5 * dt * dt
        return np.array(
            [
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def predict(self, dt):
        F = self._transition(dt)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        return float(self.state[0]), float(self.state[1])  # x, y

    def update(self, x, y, width, height):
        z = np.array([x, y])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ (z - self.H @ self.state)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.width = float(width)
        self.height = float(height)
        return float(self.state[0]), float(self.state[1])

    @property
    def kf(self):
        """Bridge gap for yolov8_node.py which expects .kf.statePost"""
        return self

    @property
    def statePost(self):
        """Bridge gap for yolov8_node.py which expects .kf.statePost"""
        return self.state


class TrackerWrapper:
    def __init__(self, tracker: ArmorPlateTracker):
        self.tracker = tracker
        self.lost_frames = 0
        self.obb_angle: float | None = None
