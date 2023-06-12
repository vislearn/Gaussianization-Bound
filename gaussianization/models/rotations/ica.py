import torch
from sklearn.decomposition import FastICA

from .lazy import LazyRotation


class ICARotation(LazyRotation):
    def fit(self, data, random_state=None):
        ica = FastICA(whiten=False, random_state=random_state)
        ica.fit(data.detach().cpu().numpy())
        transformation = ica.components_.T
        self.update_matrix(torch.from_numpy(transformation))
        return self
