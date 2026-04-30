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
        # 1) Initialize state vector: [x, y, vx, vy, ax, ay].
        self.state = np.array([x, y, 0, 0, 0, 0], dtype=np.float64)

        # 2) Configure covariance matrices.
        pn_pos, pn_vel, pn_acc = proc_noise
        self.Q = np.diag([pn_pos, pn_pos, pn_vel, pn_vel, pn_acc, pn_acc])
        self.R = np.eye(2) * meas_noise
        self.P = np.eye(6) * init_cov

        # 3) Observe x and y only.
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
        # Covariance update must mirror the corrected state to keep uncertainty bounded.
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


class UKFArmorPlateTracker:
    """5-state CTRV (Constant Turn Rate and Velocity) Unscented Kalman Filter.

    State: [x, y, v, theta, omega]
        x, y   — position (pixels)
        v      — speed magnitude (px/s)
        theta  — heading angle (rad)
        omega  — turn rate / yaw rate (rad/s)

    The nonlinear CTRV process model naturally captures curved motion
    that a linear constant-acceleration model cannot represent.
    """

    def __init__(
        self,
        x,
        y,
        width,
        height,
        proc_noise=(0.5, 0.3, 0.1),
        meas_noise=0.15,
        init_cov=0.1,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
    ):
        self.n = 5
        self.state = np.array([x, y, 0.0, 0.0, 0.0], dtype=np.float64)

        pn_pos, pn_vel, pn_yaw = proc_noise
        self.Q = np.diag([pn_pos, pn_pos, pn_vel, pn_yaw, pn_yaw])
        self.R = np.eye(2) * meas_noise
        self.P = np.eye(self.n) * init_cov

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

        self.width = float(width)
        self.height = float(height)

        self._sigmas_pred = self._sigma_points()

    def _compute_weights(self):
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n
        self._lam = lam
        c = n + lam

        self.Wm = np.full(2 * n + 1, 0.5 / c)
        self.Wc = np.full(2 * n + 1, 0.5 / c)
        self.Wm[0] = lam / c
        self.Wc[0] = lam / c + (1.0 - self.alpha**2 + self.beta)

    def _sigma_points(self):
        n = self.n
        P_stable = self.P + np.eye(n) * 1e-9
        try:
            sqrt_mat = np.linalg.cholesky((n + self._lam) * P_stable)
        except np.linalg.LinAlgError:
            self.P = np.eye(n) * 0.1
            sqrt_mat = np.linalg.cholesky((n + self._lam) * self.P)

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = self.state
        for i in range(n):
            sigmas[i + 1] = self.state + sqrt_mat[i]
            sigmas[n + i + 1] = self.state - sqrt_mat[i]
        return sigmas

    @staticmethod
    def _wrap_angle(a):
        return np.arctan2(np.sin(a), np.cos(a))

    # ---- CTRV nonlinear process model ----

    @staticmethod
    def _ctrv(s, dt):
        """Propagate one sigma point through the CTRV model."""
        x, y, v, theta, omega = s
        if abs(omega) > 1e-6:
            x_new = x + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            y_new = y + (v / omega) * (np.cos(theta) - np.cos(theta + omega * dt))
        else:
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
        return np.array([x_new, y_new, v, theta + omega * dt, omega])

    def predict(self, dt):
        sigmas = self._sigma_points()
        sigmas_pred = np.array([self._ctrv(s, dt) for s in sigmas])

        x_pred = np.dot(self.Wm, sigmas_pred)
        x_pred[3] = self._wrap_angle(x_pred[3])

        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            diff = sigmas_pred[i] - x_pred
            diff[3] = self._wrap_angle(diff[3])
            P_pred += self.Wc[i] * np.outer(diff, diff)

        self.state = x_pred
        self.P = P_pred
        self._sigmas_pred = sigmas_pred
        return float(self.state[0]), float(self.state[1])

    def update(self, x, y, width, height):
        z = np.array([x, y])
        sigmas_pred = self._sigmas_pred

        z_sigmas = sigmas_pred[:, :2]
        z_pred = np.dot(self.Wm, z_sigmas)

        S = self.R.copy()
        Pxz = np.zeros((self.n, 2))
        for i in range(2 * self.n + 1):
            z_diff = z_sigmas[i] - z_pred
            x_diff = sigmas_pred[i] - self.state
            x_diff[3] = self._wrap_angle(x_diff[3])
            S += self.Wc[i] * np.outer(z_diff, z_diff)
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)

        K = Pxz @ np.linalg.inv(S)
        self.state = self.state + K @ (z - z_pred)
        self.state[3] = self._wrap_angle(self.state[3])
        self.P = self.P - K @ S @ K.T

        self.width = float(width)
        self.height = float(height)
        return float(self.state[0]), float(self.state[1])

    @property
    def kf(self):
        return self

    @property
    def statePost(self):
        """Map CTRV [x, y, v, theta, omega] -> [x, y, vx, vy, ax, ay].

        vx, vy are decomposed from speed + heading.
        ax, ay come from the centripetal acceleration due to turning:
            a_centripetal = v * omega, directed perpendicular to heading.
        """
        x, y, v, theta, omega = self.state
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        ax = -v * omega * np.sin(theta)
        ay = v * omega * np.cos(theta)
        return np.array([x, y, vx, vy, ax, ay])


class TrackerWrapper:
    def __init__(self, tracker):
        self.tracker = tracker
        self.lost_frames = 0
        self.last_pred = None  # (px, py) from last frame's predict()
        self.obb_angle: float | None = None
