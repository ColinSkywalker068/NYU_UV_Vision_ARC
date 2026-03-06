import cv2
import numpy as np

class EuclideanTracker:
    """Fast nearest-neighbor tracking based on Euclidean distance."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = 5
        self.max_dist_sq = 15000  # ~122 pixels threshold

    def update(self, detected_centers):
        updated_tracks = {}
        unassigned = list(detected_centers)
        
        for tid, track in list(self.tracks.items()):
            best_dist_sq = float('inf')
            best_center = None
            
            for c in unassigned:
                dist_sq = (c[0] - track['center'][0])**2 + (c[1] - track['center'][1])**2
                if dist_sq < best_dist_sq and dist_sq < self.max_dist_sq:
                    best_dist_sq = dist_sq
                    best_center = c
                    
            if best_center:
                track['center'] = best_center
                track['lost'] = 0
                unassigned.remove(best_center)
            else:
                track['lost'] += 1
                
            if track['lost'] < self.max_lost:
                updated_tracks[tid] = track
                
        for c in unassigned:
            updated_tracks[self.next_id] = {'center': c, 'lost': 0}
            self.next_id += 1
            
        self.tracks = updated_tracks
        # Return only actively seen targets this frame
        return {tid: t['center'] for tid, t in self.tracks.items() if t['lost'] == 0}


class ArmorPlateTracker:
    """Original single-target Kalman Filter tracker."""
    def __init__(self, initial_x, initial_y, width=0, height=0):
        self.kf = cv2.KalmanFilter(6, 2)  # 6 states (x,y,vx,vy,ax,ay), 2 measurements (x,y)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)

        # TUNABLE PARAMETERS
        proc_noise_pos = 0.01
        proc_noise_vel = 0.05
        proc_noise_acc = 0.1

        self.kf.processNoiseCov = (np.diag([
            proc_noise_pos, proc_noise_pos, proc_noise_vel,
            proc_noise_vel, proc_noise_acc, proc_noise_acc
        ]).astype(np.float32) * 1.0)

        meas_noise = 0.15
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1

        # Initial state
        self.kf.statePost = np.array([np.float32(initial_x), np.float32(initial_y), 0, 0, 0, 0], np.float32).reshape(6, 1)

    def _update_transition_matrix(self, dt):
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

    def predict(self, dt):
        self._update_transition_matrix(dt)
        prediction_state = self.kf.predict()
        return int(prediction_state[0, 0]), int(prediction_state[1, 0])

    def update(self, x, y, width=0, height=0):
        measurement = np.array([[np.float32(x)], [np.float32(y)]], np.float32)
        self.kf.correct(measurement)


class KalmanMultiTracker:
    """Multi-target wrapper for the Kalman Filter."""
    def __init__(self):
        self.tracks = {}  # tid: ArmorPlateTracker
        self.track_lost = {} # tid: lost_frames
        self.next_id = 1
        self.max_lost = 5
        self.max_dist_sq = 15000  # Distance threshold for matching measurement to prediction
        self.dt = 1.0 / 30.0 # Assumes ~30 FPS default, robust enough for matching

    def update(self, detected_centers):
        updated_tracks = {}
        updated_lost = {}
        unassigned = list(detected_centers)
        
        # 1. Predict new locations for all existing tracks
        predictions = {}
        for tid, tracker in self.tracks.items():
            pred_x, pred_y = tracker.predict(self.dt)
            predictions[tid] = (pred_x, pred_y)

        # 2. Match measurements to predictions
        for tid, tracker in list(self.tracks.items()):
            pred_center = predictions[tid]
            best_dist_sq = float('inf')
            best_meas = None
            
            for c in unassigned:
                dist_sq = (c[0] - pred_center[0])**2 + (c[1] - pred_center[1])**2
                if dist_sq < best_dist_sq and dist_sq < self.max_dist_sq:
                    best_dist_sq = dist_sq
                    best_meas = c
                    
            if best_meas:
                tracker.update(best_meas[0], best_meas[1])
                updated_tracks[tid] = tracker
                updated_lost[tid] = 0
                unassigned.remove(best_meas)
            else:
                self.track_lost[tid] += 1
                if self.track_lost[tid] < self.max_lost:
                    updated_tracks[tid] = tracker
                    updated_lost[tid] = self.track_lost[tid]
                
        # 3. Spawn new tracks for unassigned measurements
        for c in unassigned:
            updated_tracks[self.next_id] = ArmorPlateTracker(c[0], c[1])
            updated_lost[self.next_id] = 0
            self.next_id += 1
            
        self.tracks = updated_tracks
        self.track_lost = updated_lost
        
        # Return currently visible targets with their LATEST filtered pos
        active_targets = {}
        for tid in self.tracks:
            if self.track_lost[tid] == 0:
                # Get the corrected state (x, y)
                state = self.tracks[tid].kf.statePost
                active_targets[tid] = (int(state[0, 0]), int(state[1, 0]))
                
        return active_targets
