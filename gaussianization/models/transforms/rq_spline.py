
import torch
from sinf.RQspline import (RQspline as SINFSpline, estimate_knots_gaussian)

from gaussianization.models.transforms.sinf import SINFModule


class RQSpline(SINFModule):
    def __init__(self, dims_in, bins, alpha):
        super().__init__(dims_in, SINFSpline(ndim=next(iter(dims_in))[0], nknot=bins))
        self.alpha = alpha

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        dim = next(iter(self.dims_in))[0]

        x, y, deriv = estimate_knots_gaussian(
            data,
            M=self.module.nknot,
            above_noise=torch.ones(dim, dtype=bool, device=data.device),
            derivclip=1,
            batchsize=len(data),
            alpha=self.alpha,
        )

        self.module.set_param(x, y, deriv)
