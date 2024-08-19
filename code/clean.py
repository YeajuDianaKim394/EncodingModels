"""Confound regression at run and trial level."""

import warnings

import h5py
import numpy as np
from constants import (
    CONFOUND_REGRESSORS,
    EXTRA_MOTION_CONFOUNDS,
    MOTION_CONFOUNDS,
    RUN_TRIAL_SLICE,
    RUNS,
    SUBS_STRANGERS,
    TR,
)
from nilearn import signal
from scipy.stats import zscore

# from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from util import subject
from util.path import Path

CONFOUND_MODELS = {
    "nomot": dict(run_confounds=CONFOUND_REGRESSORS),
    "runmot24": dict(
        run_confounds=CONFOUND_REGRESSORS + MOTION_CONFOUNDS + EXTRA_MOTION_CONFOUNDS
    ),
    "trialmot9": dict(
        run_confounds=CONFOUND_REGRESSORS,
        trial_confounds=MOTION_CONFOUNDS,
        add_task_confs=True,
    ),
    "splitmot6": dict(
        run_confounds=CONFOUND_REGRESSORS, split_confounds=MOTION_CONFOUNDS
    ),
}

warnings.filterwarnings("ignore", category=FutureWarning)


def main(model: str, **kwargs):

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

            if len(split_confounds):
                raise NotImplementedError

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
                    trial_pbox = prod_boxcar[i, j]

                    if add_task_confs:
                        trial_buttonP = button_idsP[i, j]
                        trial_buttonC = button_idsC[i, j]

                        conf_trial = np.hstack(
                            (
                                conf_trial,
                                trial_pbox[:, None],
                                trial_buttonP[:, None],
                                trial_buttonC[:, None],
                            )
                        )

                    # Clean the signal at the run level
                    bold_trial = signal.clean(
                        bold_trial,
                        confounds=conf_trial,
                        t_r=TR,
                        ensure_finite=True,
                        standardize="zscore_sample",
                        standardize_confounds=True,
                    )
                else:
                    bold_trial = np.nan_to_num(zscore(bold_trial))

                    # TODO: implement
                    # # if split_confounds
                    # conf_trial = zscore(conf_trial, axis=0)
                    # conf_trial = np.nan_to_num(conf_trial, copy=False)
                    # n_motconfs = conf_trial.shape[1]
                    # conf_trial2 = np.zeros((120, n_motconfs * 2 + 2))
                    # conf_trial2[trial_pmask, :n_motconfs] = conf_trial[trial_pmask]
                    # conf_trial2[~trial_pmask, n_motconfs:-2] = conf_trial[~trial_pmask]
                    # conf_trial2[:, -2] = trial_pmask.astype(conf_trial2.dtype)
                    # conf_trial2[:, -1] = (~trial_pmask).astype(conf_trial2.dtype)

                    # TODO replace this with signal.clean ?
                    # # subtract residual from actual signal
                    # reg_model = LinearRegression().fit(conf_trial2, bold_trial)
                    # # NOTE this modifies `bold` in-place
                    # bold_trial -= reg_model.predict(conf_trial2)

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

    main(**vars(parser.parse_args()))
