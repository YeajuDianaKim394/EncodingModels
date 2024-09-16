"""Confound regression at run and trial level.

for model in phys_head_task phys_head_task_split phys phys_head ; do python code/clean.py -m $model; done
for model in model9 default ; do python code/clean.py -m $model; done
for model in model9_task default_task ; do python code/clean.py -m $model; done
"""

import warnings

import h5py
import numpy as np
from constants import (
    RUN_TRIAL_SLICE,
    RUN_TRS,
    RUNS,
    SUBS_STRANGERS,
    TR,
    TRIAL_SLICES,
)
from nilearn import signal
from nilearn.glm.first_level import glover_hrf
from scipy.stats import zscore
from tqdm import tqdm
from util import subject
from util.path import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

HEAD_MOTION_CONFOUNDS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

# from Nastase et al., 2021
DEFAULT_CONFOUND_MODEL = {
    "confounds": HEAD_MOTION_CONFOUNDS + ["cosine"],
    "aCompCor": [{"n_comps": 5, "tissue": "CSF"}, {"n_comps": 5, "tissue": "WM"}],
}

# from Speer et al., 2023
CONFOUND_MODEL9 = {
    "confounds": [
        "cosine",
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        # squared
        "trans_x_power2",
        "trans_y_power2",
        "trans_z_power2",
        "rot_x_power2",
        "rot_y_power2",
        "rot_z_power2",
        # derivatives
        "trans_x_derivative1",
        "trans_y_derivative1",
        "trans_z_derivative1",
        "rot_x_derivative1",
        "rot_y_derivative1",
        "rot_z_derivative1",
        # derivative powers
        "trans_x_derivative1_power2",
        "trans_y_derivative1_power2",
        "trans_z_derivative1_power2",
        "rot_x_derivative1_power2",
        "rot_y_derivative1_power2",
        "rot_z_derivative1_power2",
        "white_matter",
        "white_matter_power2",
        "white_matter_derivative1",
        "white_matter_derivative1_power2",
        "csf",
        "csf_power2",
        "csf_derivative1",
        "csf_derivative1_power2",
    ]
}

CONFOUND_MODELS = {
    "model9": dict(confounds=CONFOUND_MODEL9),
    "default": dict(confounds=DEFAULT_CONFOUND_MODEL),
    "model9_task": dict(confounds=CONFOUND_MODEL9, add_task_confs=True),
    "default_task": dict(confounds=DEFAULT_CONFOUND_MODEL, add_task_confs=True),
}


def get_timinglog_run_regressors(dft_run):

    # create trial level boxcar
    dft_trial = dft_run[dft_run.role.str.startswith("trial").fillna(False)]
    trial_onsets = (dft_trial["run.time"] / TR).astype(int).to_numpy()
    assert len(trial_onsets) % 2 == 0
    trial_boxcar = np.zeros(RUN_TRS)
    for i in range(0, len(trial_onsets), 2):
        start, stop = trial_onsets[i], trial_onsets[i + 1]
        trial_boxcar[start:stop] = 1

    # create speaking and listening boxcars
    dft_speech = dft_run[(dft_run.role == "speaker") | (dft_run.role == "listener")]
    speech_onsets = (dft_speech["run.time"] / TR).astype(int).to_numpy()
    speech_boxcar = np.zeros(RUN_TRS)
    listen_boxcar = np.zeros(RUN_TRS)
    for i in range(len(speech_onsets) - 1):
        start = speech_onsets[i]
        stop = speech_onsets[i + 1]
        if dft_speech.iloc[i]["role"] == "speaker":
            speech_boxcar[start:stop] = 1
        else:
            listen_boxcar[start:stop] = 1

    # create button press indicators
    button_presses = np.diff(speech_boxcar, prepend=speech_boxcar[0])
    speech_buttons = np.abs(np.clip(button_presses, a_min=-1, a_max=0))
    listen_buttons = np.abs(np.clip(button_presses, a_min=0, a_max=1))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 4))
    # plt.plot(speech_boxcar)
    # plt.plot(speech_buttons)
    # plt.plot(listen_boxcar * 0.5)
    # plt.plot(listen_buttons * 0.5)
    # plt.savefig("test.png")
    # breakpoint()

    kernel = glover_hrf(TR, oversampling=1, time_length=20)

    n = len(speech_boxcar)
    task_confounds = np.stack(
        (
            np.convolve(trial_boxcar, kernel)[:n],
            np.convolve(speech_boxcar, kernel)[:n],
            np.convolve(listen_boxcar, kernel)[:n],
            np.convolve(speech_buttons, kernel)[:n],
            np.convolve(listen_buttons, kernel)[:n],
        )
    )

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 4))
    # plt.plot(task_confounds[0])
    # plt.plot(task_confounds[1])
    # plt.plot(task_confounds[2])
    # plt.plot(task_confounds[3])
    # plt.plot(task_confounds[4])
    # plt.savefig("test2.png")
    # breakpoint()

    return task_confounds.T


def run_level_regression(model: str, **kwargs):
    model_params = CONFOUND_MODELS[model]

    for sub_id in tqdm(SUBS_STRANGERS):
        dft = subject.get_timing(sub_id, condition=None)

        clean_bold = []
        run2trial = subject.get_trials(sub_id)
        for run in RUNS:

            bold = subject.get_raw_bold(sub_id, runs=[run], trial_level=False)
            bold = bold.T

            confounds = subject.get_confounds(
                sub_id,
                runs=[run],
                trial_level=False,
                model_spec=model_params["confounds"],
            )
            dft_run = dft[dft.run == run]
            if model_params.get("add_task_confs", False):
                task_confounds = get_timinglog_run_regressors(dft_run)
                confounds = np.hstack((confounds, task_confounds))

            cleaned_bold = signal.clean(
                bold,
                confounds=confounds,
                detrend=True,
                t_r=TR,
                ensure_finite=True,
                standardize="zscore_sample",
                standardize_confounds=True,
            )

            # slice out generate trials
            for trial in run2trial[run]:
                trial_slice = RUN_TRIAL_SLICE[trial]
                cleaned_bold_trial = cleaned_bold[trial_slice]
                zscored_bold_trial = np.nan_to_num(zscore(cleaned_bold_trial))
                clean_bold.append(zscored_bold_trial)

        cleaned_bold = np.vstack(clean_bold)

        boldpath = Path(
            root="data/derivatives/clean",
            datatype=model,
            sub=f"{sub_id:03d}",
            task="conv",
            space="fsaverage6",
            ext=".h5",
        )
        boldpath.mkdirs()
        with h5py.File(boldpath, "w") as f:
            f.create_dataset(name="bold", data=cleaned_bold)


def trial_level_regression(model: str, **kwargs):
    """
    three kinds of confounds: physiological, head motion, and task
    """

    model_params = CONFOUND_MODELS[model]

    kernel = glover_hrf(TR, oversampling=1, time_length=20)

    phys_regressors = model_params.get("phys_regressors", [])
    mot_regressors = model_params.get("mot_regressors", [])
    add_task_confs = model_params.get("add_task_confounds", False)
    split_confounds = model_params.get("split_confounds", False)

    for sub_id in tqdm(SUBS_STRANGERS):

        prod_boxcar, button_idsP, button_idsC = subject.get_timinglog_boxcars(sub_id)
        if add_task_confs:
            n = len(prod_boxcar)
            task_confounds = np.stack(
                (
                    np.convolve(prod_boxcar, kernel)[:n],
                    np.convolve(1 - prod_boxcar, kernel)[:n],
                    np.convolve(button_idsP, kernel)[:n],
                    np.convolve(button_idsC, kernel)[:n],
                )
            )
            task_confounds = task_confounds.T

        phys_confounds = subject.get_confounds(
            sub_id,
            trial_level=True,
            confounds=phys_regressors,
        )
        mot_confounds = subject.get_confounds(
            sub_id,
            trial_level=True,
            confounds=mot_regressors,
        )

        if split_confounds:
            prod_mask = prod_boxcar.astype(bool)
            n_motconfs = mot_confounds.shape[1]
            split_mot_confounds = np.zeros(
                (len(mot_confounds), n_motconfs * 2), dtype=mot_confounds.dtype
            )
            split_mot_confounds[prod_mask, :n_motconfs] = mot_confounds[prod_mask]
            split_mot_confounds[~prod_mask, n_motconfs:] = mot_confounds[~prod_mask]
            mot_confounds = split_mot_confounds

        bold = subject.get_raw_bold(sub_id, trial_level=True)
        bold = bold.T

        clean_bold = np.zeros_like(bold)
        for trial_slice in TRIAL_SLICES:
            bold_trial = bold[trial_slice]
            conf_trial = []

            if len(mot_regressors):
                mot_trial = mot_confounds[trial_slice]
                conf_trial.append(mot_trial)

            if add_task_confs:
                task_trial = task_confounds[trial_slice]
                conf_trial.append(task_trial)

            if len(phys_regressors):
                conf_trial.append(phys_confounds[trial_slice])

            confounds = np.hstack(conf_trial)
            cleaned_bold = signal.clean(
                bold_trial,
                confounds=confounds,
                detrend=True,
                t_r=TR,
                ensure_finite=True,
                standardize="zscore_sample",
                standardize_confounds=True,
            )

            clean_bold[trial_slice] = cleaned_bold

        cleaned_bold = np.vstack(clean_bold)

        boldpath = Path(
            root="data/derivatives/clean",
            datatype=model,
            sub=f"{sub_id:03d}",
            task="conv",
            space="fsaverage6",
            ext=".h5",
        )
        boldpath.mkdirs()
        with h5py.File(boldpath, "w") as f:
            f.create_dataset(name="bold", data=cleaned_bold)


def twostep(model: str, **kwargs):

    model_params = CONFOUND_MODELS[model]

    for sub_id in tqdm(SUBS_STRANGERS):

        prod_boxcar, button_idsP, button_idsC = subject.get_timinglog_boxcars(sub_id)
        prod_boxcar = prod_boxcar.reshape(5, 2, 120)
        button_idsP = button_idsP.reshape(5, 2, 120)
        button_idsC = button_idsC.reshape(5, 2, 120)

        run2trial = subject.get_trials(sub_id)

        cleaned_bold = []
        for i, run in enumerate(RUNS):
            bold = subject.get_raw_bold(sub_id, [run], trial_level=False)
            bold = bold.T

            run_confounds = model_params.get("run_confounds", [])
            trial_confounds = model_params.get("trial_confounds", [])
            split_confounds = model_params.get("split_confounds", [])
            add_task_confs = model_params.get("add_task_confs", False)

            if len(run_confounds):
                confounds = subject.get_confounds(
                    sub_id,
                    [run],
                    trial_level=False,
                    confounds=run_confounds,
                )

                # Clean the signal at the run level
                bold = signal.clean(
                    bold,
                    confounds=confounds,
                    t_r=TR,
                    ensure_finite=True,
                    standardize="zscore_sample",
                    standardize_confounds=True,
                )

            # Split run-level into trials
            use_trials = run2trial[run]
            for j, trial in enumerate(use_trials):

                trial_slice = RUN_TRIAL_SLICE[trial]
                bold_trial = bold[trial_slice]

                # Clean the signal at the trial level
                if len(trial_confounds):
                    confounds = subject.get_confounds(
                        sub_id,
                        [run],
                        trial_level=False,
                        confounds=trial_confounds,
                    )
                    conf_trial = confounds[trial_slice]

                    if add_task_confs:
                        trial_pbox = prod_boxcar[i, j]
                        trial_buttonP = button_idsP[i, j]
                        trial_buttonC = button_idsC[i, j]
                        conf_trial = np.hstack(
                            (
                                conf_trial,
                                trial_pbox[:, None],
                                1 - trial_pbox[:, None],
                                trial_buttonP[:, None],
                                trial_buttonC[:, None],
                            )
                        )

                    # Clean the signal at the trial level
                    bold_trial = signal.clean(
                        bold_trial,
                        confounds=conf_trial,
                        t_r=TR,
                        ensure_finite=True,
                        standardize="zscore_sample",
                        standardize_confounds=True,
                    )
                elif len(split_confounds):
                    confounds = subject.get_confounds(
                        sub_id,
                        [run],
                        trial_level=False,
                        confounds=split_confounds,
                    )
                    conf_trial = confounds[trial_slice]
                    trial_pmask = prod_boxcar[i, j].astype(bool)

                    n_motconfs = conf_trial.shape[1]
                    conf_trial2 = np.zeros(
                        (len(conf_trial), n_motconfs * 2), dtype=conf_trial.dtype
                    )
                    conf_trial2[trial_pmask, :n_motconfs] = conf_trial[trial_pmask]
                    conf_trial2[~trial_pmask, n_motconfs:] = conf_trial[~trial_pmask]

                    if add_task_confs:
                        trial_pbox = prod_boxcar[i, j]
                        trial_buttonP = button_idsP[i, j]
                        trial_buttonC = button_idsC[i, j]
                        conf_trial2 = np.hstack(
                            (
                                conf_trial2,
                                trial_pbox[:, None],
                                1 - trial_pbox[:, None],
                                trial_buttonP[:, None],
                                trial_buttonC[:, None],
                            )
                        )

                    # Clean the signal at the trial level
                    bold_trial = signal.clean(
                        bold_trial,
                        confounds=conf_trial2,
                        t_r=TR,
                        ensure_finite=True,
                        standardize="zscore_sample",
                        standardize_confounds=True,
                    )
                else:
                    bold_trial = np.nan_to_num(zscore(bold_trial))

                cleaned_bold.append(bold_trial)

        cleaned_bold = np.vstack(cleaned_bold)

        boldpath = Path(
            root="data/derivatives/clean",
            datatype=model,
            sub=f"{sub_id:03d}",
            task="conv",
            space="fsaverage6",
            ext=".h5",
        )
        boldpath.mkdirs()
        with h5py.File(boldpath, "w") as f:
            f.create_dataset(name="bold", data=cleaned_bold)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="nomot")
    parser.add_argument("-v", "--verbose", action="store_true")

    run_level_regression(**vars(parser.parse_args()))
