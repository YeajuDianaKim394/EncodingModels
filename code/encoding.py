import pickle
import warnings
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
from constants import CONFOUNDS, CONV_TRS, RUN_TRS, RUNS, TR, TRIAL_TRS
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score, correlation_score_split
from nilearn.glm.first_level import glover_hrf
from nilearn.maskers import NiftiLabelsMasker
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.atlas import get_schaefer
from util.path import Path
from voxelwise_tutorials.delayer import Delayer

print("Start", datetime.now())

parser = ArgumentParser()
parser.add_argument("-s", "--subject", type=int)
parser.add_argument("-m", "--model", type=str, default="gpt2")
parser.add_argument("-j", "--jobs", type=int, default=1)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

sub = args.subject
modelname = args.model
conv = str(sub + 100 if sub < 100 else sub)

alphas = np.logspace(-1, 8, 10)


def get_bold(
    subject: int, condition: str = "G", space: str = "MNI152NLin2009cAsym"
) -> np.ndarray:
    atlas, _ = get_schaefer(n_rois=1000)
    conv = str(sub + 100 if sub < 100 else sub)

    # Load timing path
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

    # If we want to use precise timing of trial start/end from TimingLog file
    # # keep only condition and when trials start and end
    # # we have to take line after `trial_intro` because there is the prompt screen
    # ids = sorted(
    #     (dft[dft.role == "trial_intro"].index + 1).tolist()
    #     + (dft[dft.role == "trial_end"].index).tolist()
    # )
    # dft2 = dft.iloc[ids]
    # dft2 = dft2[dft2.condition == condition]

    # load brain data
    boldpath = Path(
        root="data/derivatives/fmriprep",
        sub=f"{sub:03d}",
        ses="1",
        datatype="func",
        task="Conv",
        run=1,
        space=space,
        desc="preproc",
        suffix="bold",
        ext=".nii.gz",
    )

    # Within each run, these are the indices of each trial, excluding prompt and blanks
    trial_masks = {
        1: slice(14, 134),
        2: slice(148, 268),
        3: slice(282, 402),
        4: slice(416, 536),
    }

    # Treat trials independently
    first_trial = np.ones(8, dtype=int)
    trials = np.concatenate((first_trial, np.repeat([1, 2, 3, 4], TRIAL_TRS)))

    bold_trials = []
    for run in RUNS:
        boldpath = boldpath.update(run=run)

        confoundpath = boldpath.copy()
        del confoundpath["space"]
        confoundpath.update(desc="confounds", suffix="timeseries", ext=".tsv")
        confounds = pd.read_csv(confoundpath, sep="\t", usecols=CONFOUNDS).to_numpy()

        # # If we want to use the timings directly from TimingLog
        # trial_times = (dft2[dft2.run == run]["run.time"] / TR).round().astype(int).tolist()
        # t1onset, t1offset = trial_times[0:2]
        # t2onset, t2offset = trial_times[2:4]
        # conv_mask[trial_times[0] : trial_times[1]] = True
        # conv_mask[trial_times[2] : trial_times[3]] = True

        # Mask for conversation times
        use_trials = dft2[dft2.run == run].trial.to_numpy(dtype=int)
        conv_mask = np.zeros(RUN_TRS, dtype=bool)
        for trial in use_trials:
            conv_mask[trial_masks[trial]] = True

        # Set up masker
        masker = NiftiLabelsMasker(
            t_r=TR,
            labels_img=atlas,
            runs=trials,
            strategy="mean",
            standardize=True,
            standardize_confounds=True,
            reports=False,
            resampling_target=None,  # type: ignore
        )

        # Extract the BOLD data
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            bold = masker.fit_transform(
                boldpath,
                confounds=confounds,
                sample_mask=conv_mask,
            )
            bold_trials.append(bold)

    all_bold = np.vstack(bold_trials)
    return all_bold


def get_transcript(
    sub: Optional[int] = None, conv: Optional[str] = None, modelname: str = "gpt2"
) -> pd.DataFrame:
    if sub is None and conv is None:
        raise ValueError("Either sub or conv must be specificed.")

    if conv is None and sub is not None:
        conv = str(sub + 100 if sub < 100 else sub)

    embpath = Path(
        root="embeddings",
        conv=conv,
        datatype=modelname,
        ext=".pkl",
    )
    files = sorted(glob(embpath.starstr(["conv", "datatype"])))
    assert len(files)

    dfs = []
    for fname in files:
        df = pd.read_pickle(fname)
        dfs.append(df)
    dfemb = pd.concat(dfs).reset_index(drop=True)
    dfemb.dropna(axis=0, subset=["embedding"], inplace=True)

    # TODO ensure we have all?

    return dfemb


dfemb = get_transcript(conv=conv, modelname=modelname)

# Build regressors per TR
X = []
speaking = []
listening = []
embdim = 0
n_nuissance = 3

hrf = glover_hrf(TR, oversampling=1, time_length=32)

for (run, trial), subdf in dfemb.groupby(["run", "trial"]):
    embdim = len(subdf.iloc[0].embedding)
    nwords = np.zeros(CONV_TRS)
    embeddings = np.zeros((CONV_TRS, embdim))
    in_prod = np.zeros(CONV_TRS, dtype=bool)

    # Go through one TR at a time and find words that fall within this TR
    # average their embeddings, get num of words, etc
    for t in range(CONV_TRS):
        start_s = t * TR
        end_s = start_s + TR
        mask = (subdf.onset <= end_s) & (subdf.onset > start_s)
        if mask.any():
            nwords[t] = len(subdf)
            in_prod[t] = subdf[mask].speaker.iloc[0] == sub
            embeddings[t] = np.vstack(subdf[mask].embedding).mean(axis=0, keepdims=True)
    in_comp = ~in_prod
    speaking.append(in_prod)
    listening.append(in_comp)

    # Split embeddings into production and comprehension
    prod_embeddings = np.zeros_like(embeddings)
    comp_embeddings = np.zeros_like(embeddings)
    prod_embeddings[in_prod] = embeddings[in_prod]
    comp_embeddings[in_comp] = embeddings[in_comp]

    # Convolve prod/comp indicators
    prod_id = np.convolve(in_prod, hrf)[:CONV_TRS].reshape(-1, 1)
    comp_id = np.convolve(in_comp, hrf)[:CONV_TRS].reshape(-1, 1)
    n_words = np.convolve(nwords, hrf)[:CONV_TRS].reshape(-1, 1)
    # X_pci.append(np.stack((prod_id, comp_id, nwords), axis=1))
    x_trial = np.hstack((prod_id, comp_id, n_words, prod_embeddings, comp_embeddings))
    X.append(x_trial)

X = np.vstack(X)
speaking = np.concatenate(speaking)
listening = np.concatenate(listening)

# Set up modeling pipelines
# One for nuissance regressors with HRF
# NOTE - consider making HRF a pipeline module
preprocess_pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Kernelizer(kernel="linear"),
)

# One for embedding regressors with delay
emb_preprocess_pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Delayer(delays=[2, 3, 4, 5]),
    Kernelizer(kernel="linear"),
)

# Set up slices of design matrix into bands
feature_names = ("nuissance", "production", "comprehension")
pipelines = (preprocess_pipeline, emb_preprocess_pipeline, emb_preprocess_pipeline)
slices = [
    slice(0, n_nuissance),  # nuissance variables
    slice(n_nuissance, embdim + n_nuissance),  # production
    slice(embdim + n_nuissance, X.shape[1]),  # comprehension
]

# Make kernelizer
kernelizers_tuples = [
    (name, pipe_, slice_)
    for name, pipe_, slice_ in zip(feature_names, pipelines, slices)
]
column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=args.jobs)

params = dict(alphas=alphas, progress_bar=args.verbose)
mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
pipeline = make_pipeline(
    column_kernelizer,
    mkr_model,
)

cv_scores = []
cv_scores_prod = []
cv_scores_comp = []
cv_models = []
cv_alphas = []
cv_preds = []

run_ids = np.repeat(RUNS, CONV_TRS * 2)
kfold = PredefinedSplit(run_ids)

print("Get bold", datetime.now())
Y_bold = get_bold(sub)
print("shapes:", X.shape, Y_bold.shape)

for k, (train_index, test_index) in enumerate(kfold.split()):
    print("Fold", k, datetime.now())
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]

    pipeline = make_pipeline(column_kernelizer, mkr_model)
    pipeline["multiplekernelridgecv"].cv = PredefinedSplit(run_ids[train_index])
    pipeline.fit(X_train, Y_train)

    Y_preds = pipeline.predict(X_test, split=True)
    scores_split = correlation_score_split(Y_test, Y_preds)

    # Compute correlation based on masked prod/comp
    # TODO maybe shift by 2?
    prod_mask = speaking[test_index]
    comp_mask = ~prod_mask
    scores_prod = correlation_score(Y_test[prod_mask], Y_preds[1, prod_mask])
    scores_comp = correlation_score(Y_test[comp_mask], Y_preds[2, comp_mask])

    enc_model = pipeline["multiplekernelridgecv"]  # type: ignore
    cv_models.append(enc_model)
    cv_scores.append(scores_split)
    cv_scores_prod.append(scores_prod)
    cv_scores_comp.append(scores_comp)
    cv_alphas.append(enc_model.best_alphas_)
    cv_preds.append(Y_preds)


print("Saving", datetime.now())
result = {
    "cv_scores": cv_scores,
    "cv_scores_prod": cv_scores_prod,
    "cv_scores_comp": cv_scores_comp,
    "cv_alphas": cv_alphas,
    "cv_preds": cv_preds,
    "in_prod": speaking,
    "in_comp": listening,
    "cv_models": cv_models,
}

pklpath = Path(
    root="encoding",
    sub=f"{sub:03d}",
    datatype=modelname,
    ext=".pkl",
)
pklpath.mkdirs()
with open(pklpath.fpath, "wb") as f:
    pickle.dump(result, f)
