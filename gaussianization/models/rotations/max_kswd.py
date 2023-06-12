
from gaussianization.utils import max_kswd

from .lazy import LazyRotation


class MaxKSWDRotation(LazyRotation):
    def fit(self, data, random_state=None):
        transformation, w2_dist = max_kswd(x=data)
        self.update_matrix(transformation)
        return self
