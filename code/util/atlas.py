from os.path import isfile
from typing import Tuple

import nibabel as nib
from neuromaps.datasets import fetch_mni152
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img

DATADIR = "mats"


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
