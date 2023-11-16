import warnings
from glob import glob
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from constants import CONFOUND_REGRESSORS, RUN_TRIAL_SLICE, RUNS, TR
from nilearn import signal
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin

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

        signals = np.hstack(images)

        return signals


def get_trials(sub: str | int, condition: str = "G") -> dict:
    # Load timings. We need this to know which trials are generate condition
    timingpath = Path(
        root="stimuli",
        conv=get_conv(sub),
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
    dft2["trial"] = dft2.trial.astype(int)
    rt_dict = dft2[["run", "trial"]].groupby("run")["trial"].apply(list).to_dict()
    return rt_dict


def get_button_presses(subject: int, condition: str = "G"):
    # Load timings.
    timingpath = Path(
        root="stimuli",
        conv=get_conv(subject),
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
    dft = dft[["run", "trial", "condition", "role", "comm.time"]]
    dft = dft[dft.condition == condition]
    dft.dropna(subset=["comm.time"], inplace=True)

    # the speaker always presses. speaker == subject if subject > 100
    # a button press occurs when switching from speaker to listener for sub > 100
    # so, at the time stamps of listener start
    role = "speaker" if subject > 100 else "listener"
    press_role = "listener" if subject > 100 else "speaker"
    presses = []
    switches = []
    for _, group in dft.groupby(["run", "trial"]):
        # button press osnets
        onsets = group[group.role == press_role]["comm.time"]
        onsets = np.floor(onsets.values / TR).astype(int)
        trial_press = np.zeros(120, dtype=int)
        trial_press[onsets] = 1
        presses.append(trial_press)

        # switch onsets
        onsets = np.floor(group["comm.time"].values / TR).astype(int)
        trial_switches = np.zeros(120, dtype=int)
        s = 1 if group.iloc[0].role == role else 0
        for i in range(len(onsets) - 1):
            trial_switches[onsets[i] : onsets[i + 1]] = (i + s) % 2
        switches.append(trial_switches)

    presses = np.concatenate(presses)
    switches = np.concatenate(switches)

    return presses, switches  # , dft


def get_extra_run_confounds(subject: int | str, run: int) -> np.ndarray:
    timingpath = Path(
        root="stimuli",
        conv=get_conv(subject),
        datatype="timing",
        run=run,
        suffix="events",
        ext=".csv",
    )
    dft = pd.read_csv(timingpath)
    dft["trs"] = np.floor(dft["run.time"].values / TR).astype(int)

    # prod or comp
    in_prod = np.zeros(544, dtype=int)
    in_comp = np.zeros(544, dtype=int)
    for i, row in dft.iterrows():
        if row.role == "speaker":
            start = row.trs
            end = dft.iloc[i + 1].trs
            in_prod[start:end] = 1
        elif row.role == "listener":
            start = row.trs
            end = dft.iloc[i + 1].trs
            in_comp[start:end] = 1

    # button press osnets
    press_role = "listener" if subject > 100 else "speaker"
    onsets = dft.trs[dft.role == press_role]
    presses = np.zeros(544, dtype=int)
    presses[onsets] = 1

    conf = np.stack((in_prod, in_comp, presses)).T
    return conf


def get_bold(
    subject: int,
    condition: str = "G",
    space: str = "fsaverage6",
    confounds: list[str] = CONFOUND_REGRESSORS,
    ensure_finite: bool = True,
    return_cofounds: list[str] = [],
    use_cache: bool = False,
    save_data: bool = False,
) -> np.ndarray:
    """Return BOLD data for subject from all runs and trials."""

    cachepath = Path(
        root="data/derivatives/cleaned",
        sub=f"{subject:03d}",
        datatype="func",
        task="Conv",
        space=space,
        suffix="bold",
        ext=".npy",
    )
    cachepath.mkdirs()
    if use_cache and cachepath.isfile():
        bold = np.load(cachepath)
        if return_cofounds:
            cachepath.update(suffix="confounds")
            conf = np.load(cachepath)
            return bold, conf
        return bold

    # load brain data
    is_surface = space.startswith("fsaverage")
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

    # Array indicating each trial within this run
    # first_trial = np.ones(8, dtype=int)
    # trials = np.concatenate((first_trial, np.repeat([1, 2, 3, 4], TRIAL_TRS)))

    # Set up masker
    if space.startswith("fsaverage"):
        masker = GiftiMasker(
            t_r=TR,
            ensure_finite=True,
            standardize=True,
            standardize_confounds=True,
        )
    else:
        raise NotImplementedError()

    rt_dict = get_trials(subject)

    # Get the brain data per run, also removes confounds and applies
    bold_trials = []
    conf_trials = []
    for run in RUNS:
        boldpath = boldpath.update(run=run)

        confoundpath = boldpath.copy()
        del confoundpath["space"]
        if is_surface:
            del confoundpath["hemi"]
        confoundpath.update(desc="confounds", suffix="timeseries", ext=".tsv")
        conf_data = pd.read_csv(confoundpath, sep="\t", usecols=confounds)
        conf_data.dropna(axis=1, how="any", inplace=True)
        conf_data = conf_data.to_numpy()

        # NOTE REMEMBER THIS FOR ENCODING AND THE CACHE
        extra_confs = get_extra_run_confounds(subject, run)
        conf_data = np.hstack((conf_data, extra_confs))

        conf_ret = pd.read_csv(
            confoundpath,
            sep="\t",
            usecols=return_cofounds,
        ).to_numpy()

        boldpaths = boldpath
        if is_surface:
            boldpaths = [boldpath, boldpath.copy().update(hemi="R")]

        # Extract the BOLD data
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            bold = masker.fit_transform(
                boldpaths,  # type: ignore
                confounds=conf_data if len(confounds) else None,
            )

        # Mask for the two trials for this run that are generate condition
        use_trials = rt_dict[run]
        for trial in use_trials:
            trial_slice = RUN_TRIAL_SLICE[trial]
            bold_trials.append(zscore(bold[trial_slice], axis=0))
            conf_trials.append(conf_ret[trial_slice])

    all_bold = None
    all_bold = np.vstack(bold_trials)
    if ensure_finite:
        mask = np.logical_not(np.isfinite(all_bold))
        if mask.any():
            all_bold[mask] = 0
    if save_data:
        np.save(cachepath, all_bold)

    if len(return_cofounds):
        confs = np.vstack(conf_trials)
        if save_data:
            cachepath.update(suffix="confounds")
            conf = np.save(cachepath, confs)
        return all_bold, confs

    return all_bold


def get_transcript(
    subject: Optional[int] = None, conv: Optional[str] = None, modelname: str = "gpt2"
) -> pd.DataFrame:
    if subject is None and conv is None:
        raise ValueError("Either sub or conv must be specificed.")

    if conv is None and subject is not None:
        conv = get_conv(subject)

    embpath = Path(
        root="features",
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
