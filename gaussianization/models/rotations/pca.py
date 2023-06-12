
import torch
from sklearn.decomposition import PCA

from .lazy import LazyRotation


class PCARotation(LazyRotation):
    def fit(self, data, random_state=None):
        pca = PCA(whiten=True, svd_solver="full", random_state=random_state)
        pca.fit(data.detach().cpu().numpy())
        transformation = pca.components_.T
        self.update_matrix(torch.from_numpy(transformation))
        return self
