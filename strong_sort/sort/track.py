import numpy as np
from strong_sort.sort.kalman_filter import KalmanFilter

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, detection, track_id, class_id, conf, n_init, max_age, ema_alpha, feature=None):
        self.track_id = track_id
        self.class_id = int(class_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        self.state = TrackState.Tentative

        self.conf = conf
        self._n_init = n_init
        self._max_age = max_age
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(detection)

        self.features = []
        self.feature_buffer_size = 30  # Max feature buffer size
        if feature is not None:
            feature = feature / np.linalg.norm(feature)
            self.features.append(feature)

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection, class_id, conf):
        self.conf = conf
        self.class_id = int(class_id)
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection.to_xyah(), detection.confidence
        )

        feature = detection.feature / np.linalg.norm(detection.feature)
        if self.features:
            smooth_feat = self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
            smooth_feat /= np.linalg.norm(smooth_feat)
        else:
            smooth_feat = feature

        self.features.append(smooth_feat)
        if len(self.features) > self.feature_buffer_size:
            self.features.pop(0)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def update_without_feature(self, detection, class_id, conf):
        self.conf = conf
        self.class_id = int(class_id)
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection.to_xyah(), detection.confidence
        )
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
