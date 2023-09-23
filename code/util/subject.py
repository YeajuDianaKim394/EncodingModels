import warnings
from glob import glob
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from constants import CONFOUNDS, RUN_TRS, RUNS, TR
from nilearn import signal
from nilearn.maskers import NiftiLabelsMasker
from sklearn.base import BaseEstimator, TransformerMixin

from .atlas import get_schaefer
from .path import Path


def get_conv(subject: int) -> str:
    return str(subject + 100 if subject < 100 else subject)


def get_partner(subject: int) -> int:
    return subject + 100 if subject < 100 else subject - 100


class GiftiMasker(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.init_args = kwargs

    def fit(self, gifti_imgs: Path | list[Path], **kwargs):
        self.gifti_img = gifti_imgs
        self.init_args.update(kwargs)
        return self

    def transform(self, gifti_imgs: Path | list[Path]):
        if not isinstance(gifti_imgs, list):
            gifti_imgs = [gifti_imgs]

        images = []
        for gifti_img in gifti_imgs:
            gifti = nib.load(gifti_img)
            signals = gifti.agg_data().T  # type:ignore
            images.append(signal.clean(signals, **self.init_args))
        return np.hstack(images)


def get_bold(
    subject: int, condition: str = "G", space: str = "fsaverage6"
) -> np.ndarray:
    """Return BOLD data for subject from all runs and trials."""
    conv = get_conv(subject)

    # Load timings. We need this to know which trials are generate condition
    timingpath = Path(
        root="stimuli",
        conv=conv,
        datatype="timing",
        run=1,
        suffix="events",
        ext=".csv",
    )
    dfs = []
    for run in RUNS:
        timingpath = timingpath.update(run=run)
        dft = pd.read_csv(timingpath)
        dfs.append(dft)
    dft = pd.concat(dfs).reset_index(drop=True)
    dft2 = dft[["run", "trial", "condition"]].drop_duplicates().dropna()
    dft2 = dft2[dft2.condition == condition]

    is_surface = space.startswith("fsaverage")

    # load brain data
    boldpath = Path(
        root="data/derivatives/fmriprep",
        sub=f"{subject:03d}",
        ses="1",
        datatype="func",
        task="Conv",
        run=1,
        space=space,
        desc=None if is_surface else "preproc",
        hemi="L" if is_surface else None,
        suffix="bold",
        ext=".func.gii" if is_surface else ".nii.gz",
    )

    # Within each run, these are the indices of each trial, excluding prompt and blanks
    trial_masks = {
        1: slice(14, 134),
        2: slice(148, 268),
        3: slice(282, 402),
        4: slice(416, 536),
    }

    # Array indicating each trial within this run
    # first_trial = np.ones(8, dtype=int)
    # trials = np.concatenate((first_trial, np.repeat([1, 2, 3, 4], TRIAL_TRS)))

    # Set up masker
    if space.startswith("fsaverage"):
        masker = GiftiMasker(
            t_r=TR,
            standardize=True,
            standardize_confounds=True,
        )
    else:
        atlas, _ = get_schaefer(n_rois=1000)
        masker = NiftiLabelsMasker(
            t_r=TR,
            labels_img=atlas,
            strategy="mean",
            standardize=True,
            standardize_confounds=True,
            reports=False,
            resampling_target=None,  # type: ignore
        )

    # Get the brain data per run, also removes confounds and applies
    bold_trials = []
    for run in RUNS:
        boldpath = boldpath.update(run=run)

        confoundpath = boldpath.copy()
        del confoundpath["space"]
        if is_surface:
            del confoundpath["hemi"]
        confoundpath.update(desc="confounds", suffix="timeseries", ext=".tsv")
        confounds = pd.read_csv(confoundpath, sep="\t", usecols=CONFOUNDS).to_numpy()

        # Mask for the two trials for this run that are generate condition
        use_trials = dft2[dft2.run == run].trial.to_numpy(dtype=int)
        conv_mask = np.zeros(RUN_TRS, dtype=bool)
        for trial in use_trials:
            conv_mask[trial_masks[trial]] = True

        boldpaths = boldpath
        if is_surface:
            boldpaths = [boldpath, boldpath.copy().update(hemi="R")]

        # Extract the BOLD data
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            bold = masker.fit_transform(
                boldpaths,  # type: ignore
                confounds=confounds,
            )
            bold_trials.append(bold[conv_mask])

    all_bold = np.vstack(bold_trials)
    return all_bold


def get_transcript(
    subject: Optional[int] = None, conv: Optional[str] = None, modelname: str = "gpt2"
) -> pd.DataFrame:
    if subject is None and conv is None:
        raise ValueError("Either sub or conv must be specificed.")

    if conv is None and subject is not None:
        conv = get_conv(subject)

    embpath = Path(
        root="embeddings",
        conv=conv,
        datatype=modelname,
        ext=".pkl",
    )
    search_str = embpath.starstr(["conv", "datatype"])
    files = sorted(glob(search_str))
    if not len(files):
        raise FileNotFoundError(search_str)

    dfs = []
    for fname in files:
        df = pd.read_pickle(fname)
        tempath = Path.frompath(fname)
        df.insert(0, "trial", tempath["trial"])
        df.insert(0, "run", tempath["run"])
        dfs.append(df)
    dfemb = pd.concat(dfs).reset_index(drop=True)
    dfemb.dropna(axis=0, subset=["embedding"], inplace=True)

    dfemb["start"] = dfemb.start.interpolate(method="linear")

    return dfemb
