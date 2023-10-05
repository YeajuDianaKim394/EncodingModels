"""Utilities to make the plotting life easier
"""
from neuromaps.datasets import fetch_fsaverage
from surfplot import Plot

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
