"""Confound regression at run and trial level.

for model in phys_head_task phys_head_task_split phys phys_head ; do python code/clean.py -m $model; done
"""

import warnings

import h5py
import numpy as np
from constants import (
    RUN_TRIAL_SLICE,
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
CONFOUND_MODEL9 = [
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

CONFOUND_MODELS = {
    "model9": dict(run_confounds=CONFOUND_MODEL9),
    "model9_task": dict(run_confounds=CONFOUND_MODEL9, add_task_confs=True),

    "default": dict(run_confounds=DEFAULT_CONFOUND_MODEL),
    "default_task": dict(run_confounds=DEFAULT_CONFOUND_MODEL, add_task_confs=True),

    # "nomot": dict(run_confounds=CONFOUND_REGRESSORS),
    # "runmot": dict(run_confounds=CONFOUND_REGRESSORS + MOTION_CONFOUNDS),
    # "runmot24": dict(
    #     run_confounds=CONFOUND_REGRESSORS + MOTION_CONFOUNDS + EXTRA_MOTION_CONFOUNDS
    # ),
    # "trialmot": dict(
    #     run_confounds=CONFOUND_REGRESSORS,
    #     trial_confounds=MOTION_CONFOUNDS,
    #     add_task_confs=True,
    # ),
    # "splitmot": dict(
    #     run_confounds=CONFOUND_REGRESSORS,
    #     split_confounds=MOTION_CONFOUNDS,
    #     add_task_confs=True,
    # ),
    # # New:
    # "phys": dict(phys_regressors=CONFOUND_REGRESSORS),
    # "phys_head": dict(
    #     phys_regressors=CONFOUND_REGRESSORS, mot_regressors=MOTION_CONFOUNDS
    # ),
    # "phys_head_task": dict(
    #     phys_regressors=CONFOUND_REGRESSORS,
    #     mot_regressors=MOTION_CONFOUNDS,
    #     add_task_confounds=True,
    # ),
    # "phys_head_task_split": dict(
    #     phys_regressors=CONFOUND_REGRESSORS,
    #     mot_regressors=MOTION_CONFOUNDS,
    #     add_task_confounds=True,
    #     split_confounds=True,
    # ),
}

def run_level_regression(model: str, **kwargs):
    """
    three kinds of confounds: physiological, head motion, and task
    """

    model_params = CONFOUND_MODELS[model]

    kernel = glover_hrf(TR, oversampling=1, time_length=20)

    phys_regressors = model_params.get("phys_regressors", [])
    mot_regressors = model_params.get("mot_regressors", [])
    add_task_confs = model_params.get("add_task_confounds", False)

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

    # twostep(**vars(parser.parse_args()))
    trial_level_regression(**vars(parser.parse_args()))
