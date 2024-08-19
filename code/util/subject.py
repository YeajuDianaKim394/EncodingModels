from glob import glob

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from constants import RUN_TRIAL_SLICE, RUNS, TR
from nilearn import signal
from sklearn.base import BaseEstimator, TransformerMixin

from .path import Path


def get_conv(sub_id: int) -> str:
    return str(sub_id + 100 if sub_id < 100 else sub_id)


def get_partner(sub_id: int) -> int:
    return sub_id + 100 if sub_id < 100 else sub_id - 100


def recode_trial(trial: int) -> int:
    return ((int(trial) - 1) % 4) + 1


def get_timing(sub_id: int, condition="G") -> pd.DataFrame:
    timingpath = Path(
        root="data/stimuli",
        conv=get_conv(sub_id),
        datatype="timing",
        suffix="events",
        ext=".csv",
    )
    dft = pd.read_csv(timingpath)
    dft = dft[["run", "trial", "condition", "role", "comm.time"]]
    dft = dft[dft.condition == condition]
    dft.dropna(subset=["comm.time"], inplace=True)

    return dft


def get_trials(sub_id: int, condition: str = "G") -> dict:
    """Get which trials per run are a specific condition"""

    dft = get_timing(sub_id, condition=condition)
    dft2 = dft[["run", "trial", "condition"]].drop_duplicates().dropna()
    dft2["trial"] = dft2.trial.astype(int)
    rt_dict = dft2[["run", "trial"]].groupby("run")["trial"].apply(list).to_dict()
    return rt_dict


def get_transcript_switches(sub_id: int):
    """create a prodmask from transcript instead of timing log"""
    # prodmask impacts: clean.py when doing trial level, and encoding.

    df = get_transcript(subject=sub_id, modelname="model-gpt2-2b_layer-24")
    switches = []

    df2 = df[["run", "trial", "speaker", "start", "end"]].drop_duplicates()
    for _, group in df2.groupby(["run", "trial"]):
        group["turn"] = (group.speaker.diff() != 0).cumsum()
        group = group.groupby(["run", "trial", "turn"]).agg(
            dict(speaker="first", start="first", end="last")
        )
        group["start_tr"] = group.start / TR
        group["end_tr"] = group.end / TR

        trial_switches = np.zeros(120, dtype=int)
        for _, row in group[group.speaker == sub_id].iterrows():
            start = int(row.start_tr)
            end = int(row.end_tr)
            trial_switches[start:end] = 1

        switches.append(trial_switches)

    switches = np.concatenate(switches)
    return switches


def get_timinglog_boxcars(sub_id: int, condition: str = "G"):

    dft = get_timing(sub_id, condition=condition)

    role = "speaker" if sub_id > 100 else "listener"
    prod_boxcar = []
    for _, group in dft.groupby(["run", "trial"]):

        # switch onsets
        onsets = np.floor(group["comm.time"].values / TR).astype(int)
        trial_boxcar = np.zeros(120, dtype=int)
        s = 1 if group.iloc[0].role == role else 0
        for i in range(len(onsets) - 1):
            trial_boxcar[onsets[i] : onsets[i + 1]] = (i + s) % 2
        prod_boxcar.append(trial_boxcar)

    prod_boxcar = np.concatenate(prod_boxcar)
    button_idsA = np.clip(
        np.diff(prod_boxcar, prepend=prod_boxcar[0]), a_min=-1, a_max=0
    )
    button_idsB = np.clip(
        np.diff(prod_boxcar, prepend=prod_boxcar[0]), a_min=0, a_max=1
    )

    return prod_boxcar, button_idsA, button_idsB


def get_confounds(
    sub_id: int,
    runs: list[int] = RUNS,
    confounds: list[str] = ["framewise_displacement"],
    trial_level: bool = True,
):
    confound_path = Path(
        root="data/derivatives/fmriprep",
        sub=f"{sub_id:03d}",
        ses="1",
        datatype="func",
        task="Conv",
        run=0,
        desc="confounds",
        suffix="timeseries",
        ext=".tsv",
    )

    if trial_level:
        run2trial = get_trials(sub_id)

    dfs = []
    for run in runs:
        confound_path.update(run=run)
        conf_df = pd.read_csv(confound_path, sep="\t", usecols=confounds)
        conf_df.fillna(value=0, inplace=True)

        if trial_level:
            trials = run2trial[run]
            for trial in trials:
                trial_slice = RUN_TRIAL_SLICE[trial]
                dfs.append(conf_df.iloc[trial_slice])
        else:
            dfs.append(conf_df)

    conf_df = pd.concat(dfs)
    conf_data = conf_df.to_numpy()

    return conf_data


def get_raw_bold(
    sub_id: int,
    runs: list[int] = RUNS,
    trial_level: bool = True,
    run_time_mask=slice(None),
    voxel_mask=slice(None),
    hemisphere: str = ("L", "R"),
):
    bold_path = Path(
        root="data/derivatives/fmriprep",
        sub=f"{sub_id:03d}",
        ses="1",
        datatype="func",
        task="Conv",
        run=0,
        space="fsaverage6",
        desc=None,
        hemi="L",
        suffix="bold",
        ext=".func.gii",
    )

    if trial_level:
        run2trial = get_trials(sub_id)

    hemi_bold = []
    for hemi in hemisphere:
        bold_path.update(hemi=hemi)

        bold = []
        for run in runs:
            bold_path.update(run=run)
            img = nib.load(bold_path)
            run_bold = img.agg_data()
            run_bold = run_bold[voxel_mask, run_time_mask]

            if trial_level:
                trials = run2trial[run]
                for trial in trials:
                    trial_slice = RUN_TRIAL_SLICE[trial]
                    bold.append(run_bold[..., trial_slice])
            else:
                bold.append(run_bold)

        bold = np.hstack(bold)
        hemi_bold.append(bold)

    hemi_bold = np.vstack(hemi_bold)

    return hemi_bold


def get_bold(
    sub_id: int,
    cache: str = None,
) -> np.ndarray:
    boldpath = Path(
        root="data/derivatives/clean",
        datatype=cache,
        sub=f"{sub_id:03d}",
        task="conv",
        space="fsaverage6",
        ext=".h5",
    )
    with h5py.File(boldpath, "r") as f:
        Y_bold = f["bold"][...]

    return Y_bold


def get_transcript(sub_id: int):
    conv = get_conv(sub_id)

    transcript_path = Path(
        root="data/stimuli", conv=conv, datatype="whisperx", ext=".csv"
    )

    df = pd.read_csv(transcript_path)

    return df


def get_transcript_features(
    sub_id: int = None, modelname: str = "gpt2"
) -> pd.DataFrame:

    conv = get_conv(sub_id)

    embpath = Path(
        root="data/stimuli/",
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
