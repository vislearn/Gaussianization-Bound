from .lazy import LazyRotation


class NoRotation(LazyRotation):
    def fit(self, data, random_state=None):
        return self
