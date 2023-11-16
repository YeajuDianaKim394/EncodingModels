"""Utilities to make the plotting life easier
"""
import numpy as np
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
from neuromaps.transforms import fsaverage_to_fslr
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
from surfplot import Plot
from surfplot.utils import threshold as surf_threshold

from .atlas import Atlas


def get_surfplot(
    surface: str = "fsaverage",
    density: str = "41k",
    brightness: float = 0.7,
    sulc_alpha: float = 1.0,
    **kwargs,
) -> Plot:
    """Get a basic Plot to add layers to."""

    fetch_func = fetch_fsaverage if surface == "fsaverage" else fetch_fslr
    surfaces = fetch_func(data_dir="mats", density=density)
    surf_lh, surf_rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    p = Plot(surf_lh=surf_lh, surf_rh=surf_rh, brightness=brightness, **kwargs)
    p.add_layer(
        {"left": sulc_lh, "right": sulc_rh},
        cmap="binary_r",
        cbar=False,
        alpha=sulc_alpha,
    )
    return p


def surface_plot(
    values: np.ndarray,
    title: str = None,
    cmap: str = "coolwarm",
    cbar: bool = True,
    cbar_label: str = None,
    vmin: float = None,
    vmax: float = None,
    threshold: float = None,
    symmetric: bool = True,
    transform: str = None,
    atlas: Atlas = None,
    atlas_mode: str = "outline",
    fig=None,
    ax=None,
    **kwargs,
):
    if atlas is not None and atlas_mode == "reduce":
        values = atlas.parcellate(values)

    if vmax is None:
        vals = np.abs(values) if symmetric else values
        vmax = np.quantile(vals, 0.995)
    if vmin is None:
        vmin = 0
        if symmetric:
            vmin = -vmax
    elif vmin == "quantile":
        vals = np.abs(values) if symmetric else values
        vmin = np.quantile(vals, 0.75)

    p = get_surfplot(**kwargs)
    if threshold is not None:
        threshold = vmin if threshold == "vmin" else threshold
        values = surf_threshold(values, threshold)

    if transform == "fsaverage_to_fslr":
        n_verts = values.size // 2
        gifL = GiftiImage(darrays=(GiftiDataArray(values[:n_verts]),))
        gifR = GiftiImage(darrays=(GiftiDataArray(values[n_verts:]),))
        gifL, gifR = fsaverage_to_fslr((gifL, gifR))
        values = {"left": gifL, "right": gifR}

    p.add_layer(
        values, cmap=cmap, cbar=cbar, cbar_label=cbar_label, color_range=(vmin, vmax)
    )

    if atlas is not None and atlas_mode == "outline":
        parc_mask = atlas.label_img
        if transform == "fsaverage_to_fslr":
            n_verts = parc_mask.size // 2
            gifL = GiftiImage(darrays=(GiftiDataArray(parc_mask[:n_verts]),))
            gifR = GiftiImage(darrays=(GiftiDataArray(parc_mask[n_verts:]),))
            gifL, gifR = fsaverage_to_fslr((gifL, gifR), method="nearest")
            parc_mask = {"left": gifL, "right": gifR}
        p.add_layer(parc_mask, cmap="Greys", as_outline=True, cbar=False, alpha=0.8)

    if fig is None and ax is None:
        fig = p.build()
        if title is not None:
            fig.suptitle(title)
    else:
        # copied from source code of Plot.build() so i can use my own fig/ax.
        plotter = p.render()
        plotter._check_offscreen()
        x = plotter.to_numpy(transparent_bg=True, scale=(2, 2))

        if ax is None:
            figsize = tuple((np.array(p.size) / 100) + 1)
            ax = fig.subplots(figsize=figsize)

        ax.imshow(x)
        ax.axis("off")

        if cbar:
            p._add_colorbars()

        if title is not None:
            ax.set_title(title + f" ({vmin:.3f}, {vmax:.3f})")

    return fig
