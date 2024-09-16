"""Utilities to make the plotting life easier
"""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from brainspace.mesh.mesh_io import read_surface
from matplotlib.colors import Normalize
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
from neuromaps.transforms import (
    fsaverage_to_fsaverage,
    fsaverage_to_fslr,
    mni152_to_fsaverage,
)
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
from surfplot import Plot
from surfplot.utils import threshold as surf_threshold

from .atlas import Atlas

_image_cache = {}


def two_brain_fig(**kwargs):
    return plt.subplots(1, 2, figsize=(4.5, 4.5), **kwargs)


def standalone_colorbar(
    cmap: str,
    ticks=(0, 1),
    tick_labels=(0, 1),
    figsize=(1.5, 1),
    orientation="h",
    dpi: int = 300,
    font_size: float = 8,
):
    """Adapted from https://stackoverflow.com/a/62436015"""

    if orientation.startswith("h"):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # left, bottom, width, height
        # fractions of figure width and height
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
        cbar = mpl.colorbar.ColorbarBase(ax, orientation="horizontal", cmap=cmap)
        cbar.ax.set_xticks(ticks)
        cbar.ax.set_xticklabels(tick_labels)
    else:
        fig = plt.figure(figsize=(figsize[1], figsize[0]), dpi=dpi)
        ax = fig.add_axes([0.05, 0.80, 0.1, 0.9])
        cbar = mpl.colorbar.ColorbarBase(ax, orientation="vertical", cmap=cmap)
        cbar.ax.set_yticks(ticks)
        cbar.ax.set_yticklabels(tick_labels)

    cbar.ax.tick_params(labelsize=font_size)

    return fig


def upsample_fsaverage(values: np.ndarray, method: str = "linear") -> np.ndarray:
    dataL = values[:40962]
    dataR = values[40962:]
    gifL = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(dataL),))
    gifR = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(dataR),))
    gifLn, gifRn = fsaverage_to_fsaverage(
        (gifL, gifR), "164k", hemi=("L", "R"), method=method
    )
    resampled_data = np.concatenate((gifLn.agg_data(), gifRn.agg_data()))
    return resampled_data


def get_surfplot(
    surface: str = "fsaverage",
    density: str = "41k",
    brightness: float = 0.7,
    sulc_alpha: float = 0.5,
    add_sulc: bool = False,
    surf_lh_fn: str = "mats/suma-fsaverage6/lh.inf_{}.gii",
    surf_rh_fn: str = "mats/suma-fsaverage6/rh.inf_{}.gii",
    inflation: int = 100,
    **kwargs,
) -> Plot:
    """Get a basic Plot to add layers to."""

    fetch_func = fetch_fsaverage if surface == "fsaverage" else fetch_fslr
    surfaces = fetch_func(data_dir="mats", density=density)
    if surf_lh_fn is None or surf_rh_fn is None:
        surf_lh_fn, surf_rh_fn = surfaces["inflated"]
    else:
        surf_lh_fn = surf_lh_fn.format(inflation)
        surf_rh_fn = surf_rh_fn.format(inflation)
    sulc_lh_fn, sulc_rh_fn = surfaces["sulc"]

    if surf_lh_fn not in _image_cache:
        _image_cache[surf_lh_fn] = read_surface(str(surf_lh_fn))
    if surf_rh_fn not in _image_cache:
        _image_cache[surf_rh_fn] = read_surface(str(surf_rh_fn))
    if sulc_lh_fn not in _image_cache:
        _image_cache[sulc_lh_fn] = nib.load(str(sulc_lh_fn))
    if sulc_rh_fn not in _image_cache:
        _image_cache[sulc_rh_fn] = nib.load(str(sulc_rh_fn))

    surf_lh = _image_cache[surf_lh_fn]
    surf_rh = _image_cache[surf_rh_fn]
    sulc_lh = _image_cache[sulc_lh_fn]
    sulc_rh = _image_cache[sulc_rh_fn]

    p = Plot(surf_lh=surf_lh, surf_rh=surf_rh, brightness=brightness, **kwargs)
    if add_sulc:
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
    zeroNan=True,
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
        if isinstance(threshold, float):
            threshold = vmin if threshold == "vmin" else threshold
            values = surf_threshold(values, threshold)
        elif isinstance(threshold, np.ndarray):
            values = values.copy()
            values[threshold] = 0

    if transform == "fsaverage_to_fslr":
        n_verts = values.size // 2
        gifL = GiftiImage(darrays=(GiftiDataArray(values[:n_verts]),))
        gifR = GiftiImage(darrays=(GiftiDataArray(values[n_verts:]),))
        gifL, gifR = fsaverage_to_fslr((gifL, gifR))
        values = {"left": gifL, "right": gifR}

    p.add_layer(
        values,
        cmap=cmap,
        cbar=cbar,
        cbar_label=cbar_label,
        color_range=(vmin, vmax),
        zero_transparent=zeroNan,
    )

    if atlas is not None and atlas_mode == "outline":
        parc_mask = atlas.label_img
        if transform == "fsaverage_to_fslr":
            n_verts = parc_mask.size // 2
            gifL = GiftiImage(darrays=(GiftiDataArray(parc_mask[:n_verts]),))
            gifR = GiftiImage(darrays=(GiftiDataArray(parc_mask[n_verts:]),))
            gifL, gifR = fsaverage_to_fslr((gifL, gifR), method="nearest")
            parc_mask = {"left": gifL, "right": gifR}
        p.add_layer(parc_mask, cmap="gray", as_outline=True, cbar=False)

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
            p._add_colorbars(fig=fig, ax=ax, n_ticks=2)

        if title is not None:
            ax.set_title(title)

    return fig


def get_surf_grad(
    axis: str = "transverse",
    vmin: float = 0.0,
    vmax: float = 1.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Get a gradient in fsaverage space.
    # coronal: inferior - superior
    # transverse: posterior - anterior
    # saggital: lateral - medial
    """
    gradimg = np.zeros((91, 109, 91), dtype=dtype)

    # post-ant
    if axis == "transverse":
        grad = np.linspace(vmin, vmax, 109, dtype=gradimg.dtype)
        for i, j in np.ndindex(91, 91):
            gradimg[i, :, j] = grad
    # ventral-dorsal
    elif axis == "coronal":
        grad = np.linspace(vmin, vmax, 91, dtype=gradimg.dtype)
        for i, j in np.ndindex(91, 109):
            gradimg[i, j, :] = grad
    # lateral-medial
    elif axis == "saggital":
        grad = np.linspace(vmin, vmax, 91, dtype=gradimg.dtype)
        for i, j in np.ndindex(109, 91):
            gradimg[:, i, j] = grad

    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    new_nimg = nib.nifti1.Nifti1Image(gradimg, affine=affine)

    gifL, gifR = mni152_to_fsaverage(new_nimg, method="linear")
    fs_grad = np.concatenate((gifL.agg_data(), gifR.agg_data()))
    return fs_grad


class Colormap2D(mcolors.Colormap):
    def __init__(
        self, cmap="PU_BuOr_covar", vmin=None, vmax=None, vmin2=None, vmax2=None
    ):
        img = plt.imread(f"mats/{cmap}.png")
        self.cmap = mcolors.ListedColormap(np.squeeze(img))
        self.vmin = vmin
        self.vmax = vmax
        self.vmin2 = vmin if vmin2 is None else vmin2
        self.vmax2 = vmax if vmax2 is None else vmax2
        N = self.cmap.colors.shape[0]
        super().__init__(cmap, N)

    def __call__(self, X, alpha=None, bytes=False):
        if X.ndim == 1:
            data1 = X
            data2 = X
        elif X.ndim == 2:
            data1 = X[:, 0]
            data2 = X[:, 1]

        cmap = self.cmap.colors

        norm1 = Normalize(self.vmin, self.vmax)
        norm2 = Normalize(self.vmin2, self.vmax2)

        d1 = np.clip(norm1(data1), 0, 1)
        d2 = np.clip(1 - norm2(data2), 0, 1)
        dim1 = np.round(d1 * (cmap.shape[1] - 1))
        # Nans in data seemed to cause weird interaction with conversion to uint32
        dim1 = np.nan_to_num(dim1).astype(np.uint32)
        dim2 = np.round(d2 * (cmap.shape[0] - 1))
        dim2 = np.nan_to_num(dim2).astype(np.uint32)

        colored = cmap[dim2.ravel(), dim1.ravel()]
        # map r, g, b, a values between 0 and 255 to avoid problems with
        # VolumeRGB when plotting flatmaps with quickflat
        colored = (colored * 255).astype(np.uint8)
        r, g, b, a = colored.T
        r.shape = dim1.shape
        g.shape = dim1.shape
        b.shape = dim1.shape
        a.shape = dim1.shape
        # Preserve nan values as alpha = 0
        aidx = np.logical_or(np.isnan(data1), np.isnan(data2))
        a[aidx] = 0
        return r, g, b, a

    def __hash__(self):
        return hash(self.name)
