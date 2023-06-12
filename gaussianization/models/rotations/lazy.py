
import torch
from FrEIA.modules import FixedLinearTransform


class LazyRotation(FixedLinearTransform):
    def __init__(self, dims_in):
        dims_in = list(dims_in)
        if len(dims_in) > 1:
            raise ValueError(f"Can only handle single input, found {len(dims_in)}.")
        if len(dims_in[0]) > 1:
            raise ValueError(f"Can only handle vectors, found shape {dims_in[0]}.")
        super().__init__(dims_in, M=torch.eye(dims_in[0][0]))

    def update_matrix(self, matrix):
        # Keep device
        matrix = matrix.to(self.M)

        self.M.data = matrix.t()
        self.M_inv.data = matrix.t().inverse()
        self.logDetM.data = torch.slogdet(matrix)[1]

    @torch.no_grad()
    def fit(self, data, random_state=None):
        raise NotImplementedError
