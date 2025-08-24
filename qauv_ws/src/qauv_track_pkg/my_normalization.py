import numpy as np
import joblib


class ObsActNormalizer:
    def __init__(self):
        self.obs_mean = None
        self.obs_std  = None
        self.act_mean = None
        self.act_std  = None

    def fit(self, obs, acts):
        self.obs_mean = obs.mean(axis=0)
        self.obs_std  = obs.std(axis=0) + 1e-8
        self.act_mean = acts.mean(axis=0)
        self.act_std  = acts.std(axis=0) + 1e-8

    def normalize_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return (obs - self.obs_mean) / self.obs_std

    def normalize_act(self, act):
        act = np.asarray(act, dtype=np.float32)
        return (act - self.act_mean) / self.act_std

    def denormalize_act(self, act_norm):
        act_norm = np.asarray(act_norm, dtype=np.float32)
        return act_norm * self.act_std + self.act_mean

    def save(self, path):
        joblib.dump(vars(self), path)

    @classmethod
    def load(cls, path):
        norm = cls()
        for k, v in joblib.load(path).items():
            setattr(norm, k, v)
        return norm
