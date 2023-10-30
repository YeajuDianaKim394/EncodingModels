"""Utilities to make the plotting life easier
"""
import numpy as np
from neuromaps.datasets import fetch_fsaverage
from surfplot import Plot

from .atlas import Atlas

SURFS = fetch_fsaverage(data_dir="mats", density="41k")
SURF_LH, SURF_RH = SURFS["inflated"]
SULC_LH, SULC_RH = SURFS["sulc"]


def get_surfplot(**kwargs) -> Plot:
    """Get a basic Plot to add layers to."""
    p = Plot(surf_lh=SURF_LH, surf_rh=SURF_RH, brightness=0.7, **kwargs)
    p.add_layer(
        {"left": SULC_LH, "right": SULC_RH}, cmap="binary_r", cbar=False, alpha=0.5
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
    symmetric: bool = True,
    atlas: Atlas = None,
    atlas_mode: str = "outline",
):
    if atlas is not None and atlas_mode == "reduce":
        values = atlas.parcellate(values)

    if vmax is None:
        vmax = np.quantile(values, 0.999)
    if vmin is None:
        vmin = 0
        if symmetric:
            vmin = -vmax

    p = get_surfplot()
    p.add_layer(
        values, cmap=cmap, cbar=cbar, cbar_label=cbar_label, color_range=(vmin, vmax)
    )

    if atlas is not None and atlas_mode == "outline":
        parc_mask = atlas.label_img
        p.add_layer(parc_mask, cmap="gray", as_outline=True, cbar=False, alpha=0.8)

    fig = p.build()
    if title is not None:
        fig.suptitle(title)
    return fig
