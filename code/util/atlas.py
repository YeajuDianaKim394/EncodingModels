from os.path import isfile
from typing import Tuple

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img

DATADIR = "mats"


def reduce_voxels(
    values: np.ndarray,
    parcellation: np.ndarray = None,
    agg_func=np.mean,
    start_label: int = 1,
) -> np.ndarray:
    """Reduce voxel-wise data into parcel-level data."""
    if parcellation is None:
        parcellation, _ = get_glasser()

    n_parcels = np.unique(parcellation).size
    parcel_values = np.zeros_like(n_parcels - start_label)
    for i in range(n_parcels):
        parcel_mask = parcellation == (i + start_label)
        parcel_values[i] = agg_func(values[parcel_mask])

    return parcel_values


def expand_parcels(
    values: np.ndarray,
    parcellation: np.ndarray = None,
    agg_func=np.mean,
    start_label: int = 1,
) -> np.ndarray:
    """Expand parcel-level values to voxel-level.

    Assigns a voxels' value equal to the parcel value.
    """
    if parcellation is None:
        parcellation, _ = get_glasser()

    n_parcels = np.unique(parcellation).size
    voxel_values = np.zeros_like(parcellation)
    for i in range(n_parcels):
        parcel_mask = parcellation == (i + start_label)
        voxel_values[parcel_mask] = values[i]

    return voxel_values


def parcellate_voxels(
    values: np.ndarray, parcellation: np.ndarray = None, start_label: int = 1
) -> np.ndarray:
    """Aggregate voxel-wise data into parcel-level.

    Assigns a voxel's data point equal to the mean of its parcel.
    """
    if parcellation is None:
        parcellation, _ = get_glasser()

    n_parcels = np.unique(parcellation).size
    parcel_values = np.zeros_like(values)
    for i in range(start_label, n_parcels):
        parcel_mask = parcellation == i
        parcel_values[parcel_mask] = values[parcel_mask].mean()

    return parcel_values


def get_glasser() -> (np.ndarray, dict):
    """Get the Glasser parcellation labels."""
    dsegL = nib.load(f"{DATADIR}/tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii")
    dsegR = nib.load(f"{DATADIR}/tpl-fsaverage6_hemi-R_desc-MMP_dseg.label.gii")
    lh_parc = dsegL.agg_data()
    rh_parc = dsegR.agg_data()

    parc_mask = np.hstack((lh_parc, rh_parc))
    id2label = dsegL.labeltable.get_labels_as_dict()

    return parc_mask, id2label


def get_schaefer(
    n_rois: int = 1000,
    n_networks: int = 17,
    density: int = 3,
    force_resample: bool = False,
) -> Tuple[str, list[str]]:
    """Get the schaefer atlas resampled to our MNI space and resolution."""
    filepath = (
        f"{DATADIR}/Schaefer2018_{n_rois}Parcels_{n_networks}Networks"
        f"_order_FSLMNI152_{density}mm.nii.gz"
    )

    # Get the requested schaefer atlas
    schaefer = fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=n_networks,
        resolution_mm=1,
        data_dir=DATADIR,
    )
    labels = [s.decode() for s in schaefer["labels"].tolist()]

    if not isfile(filepath) or force_resample:
        # Get MNI mask
        # mni_atlases = fetch_mni152(density=f"{density}mm", data_dir=DATADIR)
        # brain_mask = mni_atlases["2009cAsym_brainmask"]
        brain_mask = (
            "data/derivatives/fmriprep/sub-004/ses-1/func/sub-004_ses-1_task-"
            "Conv_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )

        # Resample schaefer to MNI
        schaefer_atlas = schaefer["maps"]
        resampled = resample_to_img(schaefer_atlas, brain_mask, interpolation="nearest")
        nib.save(resampled, filepath)

    return filepath, labels
