#!/usr/bin/env python

# PU_RdBu_covar PU_RdBu_covar_alpha
# , cmap='PU_RdBu_covar', vmin=0.05, vmin2=0.05, vmax=.15, vmax2=.15)

import argparse

import cortex
import h5py

panels = cortex.export.params_flatmap_lateral_medial
# cortex.utils.download_subject(subject_id='fsaverage')#, download_again=True)

parser = argparse.ArgumentParser()
parser.add_argument("keys", nargs="+")
parser.add_argument("--cmap", type=str, default=None)
parser.add_argument("--vmin", type=float, default=None)
parser.add_argument("--vmax", type=float, default=None)
parser.add_argument("--merge", action="store_true")
parser.add_argument("--webshow", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--recache", action="store_true")
args = parser.parse_args()

if args.merge:
    assert len(args.keys)
    assert len(args.keys) == 2

cmap_args = dict(cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)

viewer_params = dict(
    overlays_visible=("rois", "sulci"),
    labels_visible=("rois",),
    recache=args.recache,
    overlay_file="fsaverage-overlays-glasser.svg",
)

data = {}
with h5py.File("fig2-encoding.hdf5", "r") as f:
    for key in args.keys:
        data[key] = f[key][...]

if args.merge:
    values1 = data[args.keys[0]]
    values2 = data[args.keys[1]]
    volume = cortex.Vertex2D(values1, values2, "fsaverage")
elif len(args.keys) > 1:
    volume = {key: cortex.Vertex(data[key], "fsaverage") for key in args.keys}
else:
    volume = cortex.Vertex(data[key], "fsaverage", **cmap_args)

if args.webshow:
    viewer = cortex.webshow(volume, **viewer_params)
    input("Waiting for browser to close")
    # breakpoint()

if args.save:
    fig = cortex.export.plot_panels(
        volume, **panels, save_name="output.png", viewer_params=viewer_params
    )
