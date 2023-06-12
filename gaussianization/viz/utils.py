import matplotlib.pyplot as plt
from functools import wraps
from typing import Sized
from math import ceil


def subplot_grid(count, ncols=3, expand=True, width_each=None, height_each=None,
                 figsize=None, **kwargs):
    if ncols is None:
        ncols = count
    if count < ncols:
        ncols = count
    nrows = ceil(count / ncols)

    if figsize is None:
        if width_each is None:
            width_each = 4
        if height_each is None:
            height_each = 4
        kwargs["figsize"] = (ncols * width_each, nrows * height_each)
    else:
        assert width_each is None, "Cannot give both figsize and width"
        assert height_each is None, "Cannot give both figsize and height"
        kwargs["figsize"] = figsize

    if nrows * ncols == 0:
        return None, None
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **kwargs)
    if count % ncols != 0:
        for which in range(count % ncols - ncols, 0):
            axes[-1][which].axis("off")
    return fig, axes.reshape(-1)[:count] if expand else axes


@wraps(subplot_grid)
def iter_ax_grid(sized: Sized, *args, **kwargs):
    fig, axes = subplot_grid(len(sized), *args, **kwargs)
    for ax, item in zip(axes, sized):
        plt.sca(ax)
        yield item
