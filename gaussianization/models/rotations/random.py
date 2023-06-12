
import torch
from scipy.stats import ortho_group
from .lazy import LazyRotation


class RandomRotation(LazyRotation):
    def fit(self, data, random_state=None):
        transformation = ortho_group.rvs(data.shape[1], random_state=random_state)
        self.update_matrix(torch.from_numpy(transformation))
        return self
