"""Utilities to make the plotting life easier
"""
import nibabel as nib
import numpy as np
from brainspace.mesh.mesh_io import read_surface
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
from neuromaps.transforms import fsaverage_to_fslr, mni152_to_fsaverage
from neuromaps.transforms import fsaverage_to_fsaverage
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
from surfplot import Plot
from surfplot.utils import threshold as surf_threshold

from .atlas import Atlas

_image_cache = {}

def upsample_fsaverage(values: np.ndarray, method: str = 'linear') -> np.ndarray:
    dataL = values[:40962]
    dataR = values[40962:]
    gifL = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(dataL),))
    gifR = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(dataR),))
    gifLn, gifRn = fsaverage_to_fsaverage((gifL, gifR), '164k', hemi=('L', 'R'), method=method)
    resampled_data = np.concatenate((gifLn.agg_data(), gifRn.agg_data()))
    return resampled_data

def get_surfplot(
    surface: str = "fsaverage",
    density: str = "41k",
    brightness: float = 0.7,
    sulc_alpha: float = 0.5,
    **kwargs,
) -> Plot:
    """Get a basic Plot to add layers to."""

    fetch_func = fetch_fsaverage if surface == "fsaverage" else fetch_fslr
    surfaces = fetch_func(data_dir="mats", density=density)
    surf_lh_fn, surf_rh_fn = surfaces["inflated"]
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
