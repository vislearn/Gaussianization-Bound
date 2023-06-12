import numpy as np
import matplotlib.pyplot as plt
import torch


MESH_MODE_MESH = "mesh"
MESH_MODE_CONTOUR = "contour"
MESH_MODE_CONTOUR_FILLED = "contourf"
MESH_MODE_RETURN_GRID = "return_grid"


def plot_mesh_in_mode(mode, x, y, values, levels=None, vmin=None, vmax=None,
                      aspect="equal", shading="auto", symmetric=False,
                      symmetric_around=0, no_maxn_levels=False, **kwargs):
    if levels is not None:
        try:
            iter(levels)
            if vmin is None:
                vmin = np.min(levels)
            if vmax is None:
                vmax = np.max(levels)
        except TypeError:
            pass
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    if no_maxn_levels and isinstance(levels, int):
        levels = np.linspace(vmin, vmax, levels)
    if symmetric:
        max_from_center = max(abs(symmetric_around - vmin),
                              abs(symmetric_around - vmax))
        vmin = symmetric_around - max_from_center
        vmax = symmetric_around + max_from_center
    if mode == MESH_MODE_MESH:
        result = plt.pcolormesh(x, y, values, vmin=vmin, vmax=vmax,
                                shading=shading, **kwargs)
    elif mode == MESH_MODE_CONTOUR:
        result = plt.contour(x, y, values, levels=levels, vmin=vmin, vmax=vmax,
                             **kwargs)
    elif mode == MESH_MODE_CONTOUR_FILLED:
        result = plt.contourf(x, y, values, levels=levels, vmin=vmin, vmax=vmax,
                              **kwargs)
    elif mode == MESH_MODE_RETURN_GRID:
        return values
    else:
        raise ValueError(f"Mesh mode {mode}")
    plt.gca().set_aspect(aspect)
    return result


@torch.no_grad()
def density_mesh(density, pos_min, pos_max, resolution=250, vmin=None,
                 vmax=None, fallback=True, log=False, aspect="equal",
                 mesh_mode=MESH_MODE_MESH, dtype=None, device=None,
                 fallback_sample_factor=100, **kwargs):
    # pos_min, pos_max = ensure_bounds(density, pos_min, pos_max)
    x, y, pos = build_mesh(pos_min, pos_max, resolution, device=device,
                           dtype=dtype)

    try:
        if fallback == "force":
            raise NotImplementedError()
        den = density.log_prob(pos)
        if not log:
            den.exp_()
    except NotImplementedError:
        if not fallback:
            raise

        samples = density.sample((
            fallback_sample_factor * resolution ** 2,)).numpy()
        pos_min = ensure_list(pos_min, 2)
        pos_max = ensure_list(pos_max, 2)
        den, *_ = np.histogram2d(samples[:, 1], samples[:, 0], resolution,
                                 [[pos_min[1], pos_max[1]],
                                  [pos_min[0], pos_max[0]]], density=True)
        if log:
            den = torch.log(den)

    return plot_mesh_in_mode(mesh_mode, x.cpu(), y.cpu(), den.reshape(x.shape).cpu(),
                             vmin=vmin, vmax=vmax, aspect=aspect, **kwargs)


def ensure_list(value, dim):
    if isinstance(value, (int, float)):
        value = [value] * dim
    else:
        assert len(value) == dim
    return value


def build_mesh(pos_min, pos_max, resolution, dtype=None, device=None):
    pos_min = ensure_list(pos_min, 2)
    pos_max = ensure_list(pos_max, 2)
    resolution = ensure_list(resolution, 2)
    lin_spaces = [torch.linspace(a, b, r) for a, b, r in zip(pos_min, pos_max, resolution)]
    grid = [x.T for x in torch.meshgrid(lin_spaces)]
    pos = torch.stack(grid, 2).reshape(-1, 2)

    return *[g.to(device, dtype) for g in grid], pos.to(device, dtype)

