import FrEIA
import torch


class SINFModule(FrEIA.modules.InvertibleModule):
    def __init__(self, dims_in, module):
        super().__init__(dims_in)
        self.module = module

    def forward(self, x_or_z, c=None, rev: bool = False, jac: bool = True):
        if not rev:
            z_or_x, jac = self.module.forward(x_or_z[0])
        else:
            z_or_x, jac = self.module.inverse(x_or_z[0])
            jac = -jac
        jac = torch.sum(jac, dim=-1)
        return (z_or_x,), jac

    def output_dims(self, input_dims):
        return input_dims
