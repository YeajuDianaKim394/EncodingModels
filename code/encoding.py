import pickle
from argparse import ArgumentParser
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from constants import NRUNS, RUNS, TR
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score_split
from nilearn import datasets
from nilearn.glm.first_level import glover_hrf
from nilearn.maskers import NiftiLabelsMasker
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
confounds = [
    "a_comp_cor_00",
    "a_comp_cor_01",
    "a_comp_cor_02",
    "a_comp_cor_03",
    "a_comp_cor_04",
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "cosine00",
]


eventpath = Path(
    root="stimuli",
    conv=conv,
    datatype="timing",
    run=1,
    suffix="events",
    ext=".csv",
)

dfs = []
for run in RUNS:
    eventpath = eventpath.update(run=run)
    dft = pd.read_csv(eventpath)
    dfs.append(dft)
dft = pd.concat(dfs).reset_index(drop=True)

# keep only condition and when trials start and end
# we have to take line after `trial_intro` because there is the prompt screen
ids = sorted(
    (dft[dft.role == "trial_intro"].index + 1).tolist()
    + (dft[dft.role == "trial_end"].index).tolist()
)
dft2 = dft.iloc[ids]
dft2 = dft2[dft2.condition == "G"]


# load brain data

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2)

boldpath = Path(
    root="data/derivatives/fmriprep",
    sub=f"{sub:03d}",
    ses="1",
    datatype="func",
    task="Conv",
    run=1,
    space="MNI152NLin2009cAsym",
    desc="preproc",
    suffix="bold",
    ext=".nii.gz",
)

sub_maskers = []
bold_trials = []
for run in RUNS:
    boldpath = boldpath.update(run=run)
    print(run, boldpath.fpath)

    confoundpath = boldpath.copy()
    del confoundpath["space"]
    confoundpath.update(desc="confounds", suffix="timeseries", ext=".tsv")
    confound_df = pd.read_csv(confoundpath, sep="\t", usecols=confounds)

    # Resample
    label_masker = NiftiLabelsMasker(labels_img=atlas.maps)
    fmri_matrix = label_masker.fit_transform(boldpath, confounds=confound_df.to_numpy())
    sub_maskers.append(label_masker)

    trial_times = (dft2[dft2.run == run]["run.time"] / TR).round().astype(int).tolist()
    t1onset, t1offset = trial_times[0:2]
    t2onset, t2offset = trial_times[2:4]

    bold_trials.append(fmri_matrix[t1onset:t1offset])
    bold_trials.append(fmri_matrix[t2onset:t2offset])

Y_bold = np.vstack(bold_trials)

# Load embeddings for all runs and trials (as available)
embpath = Path(
    root="embeddings",
    conv=conv,
    datatype=modelname,
    # suffix="transcript",
    ext=".pkl",
)
files = sorted(glob(embpath.starstr(["conv", "datatype"])))
assert len(files)
dfs = []
for fname in files:
    df = pd.read_pickle(fname)
    dfs.append(df)
dfemb = pd.concat(dfs).reset_index(drop=True)
dfemb.dropna(axis=0, subset=['embedding'], inplace=True)
print(dfemb.shape)

# Build regressors per TR
stim_trs = 120

X_emb = []
X_pci = []

for (run, trial), subdf in dfemb.groupby(["run", "trial"]):
    print(run, trial, subdf.shape)

    dims = len(subdf.iloc[0].embedding)
    nwords = np.zeros(stim_trs)
    embeddings = np.zeros((stim_trs, dims))
    in_prod = np.zeros(stim_trs, dtype=bool)

    # Go through one TR at a time and find words that fall within this TR
    # average their embeddings, get num of words, etc
    for t in range(stim_trs):
        start_s = t * TR
        end_s = start_s + TR
        mask = (subdf.onset <= end_s) & (subdf.onset > start_s)
        if mask.any():
            nwords[t] = len(subdf)
            in_prod[t] = subdf[mask].speaker.iloc[0] == sub
            embeddings[t] = np.vstack(subdf[mask].embedding).mean(axis=0, keepdims=True)
    in_comp = ~in_prod

    # Split embeddings into production and comprehension
    prod_embeddings = np.zeros_like(embeddings)
    comp_embeddings = np.zeros_like(embeddings)
    prod_embeddings[in_prod] = embeddings[in_prod]
    comp_embeddings[in_comp] = embeddings[in_comp]
    X_emb.append(np.hstack((prod_embeddings, comp_embeddings)))

    # Convolve prod/comp indicators
    hrf = glover_hrf(TR, oversampling=TR, time_length=32)
    prod_id = np.convolve(in_prod, hrf, mode="same")
    comp_id = np.convolve(in_comp, hrf, mode="same")
    X_pci.append(np.stack((prod_id, comp_id), axis=1))

X_emb = np.vstack(X_emb)
X_pci = np.vstack(X_pci)


# Modeling
print("On to modeling")

preprocess_pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Kernelizer(kernel="linear"),
)

emb_preprocess_pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    Delayer(delays=[1, 2, 3, 4]),
    Kernelizer(kernel="linear"),
)

feature_names = ("pci", "prod", "comp")
pipelines = (preprocess_pipeline, emb_preprocess_pipeline, emb_preprocess_pipeline)
slices = [slice(0, 2), slice(2, dims + 2), slice(dims + 2, None)]


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
cv_models = []
cv_alphas = []

run_ids = np.repeat(np.arange(NRUNS), stim_trs * 2)
kfold = PredefinedSplit(run_ids)

print("shapes", X_emb.shape, X_pci.shape, Y_bold.shape)

for k, (train_index, test_index) in enumerate(kfold.split()):
    print("Fold", k, datetime.now())
    X_emb_train = X_emb[train_index]
    X_pci_train = X_pci[train_index]
    X_train = np.hstack((X_pci_train, X_emb_train))
    X_test = np.hstack((X_pci[test_index], X_emb[test_index]))

    Y_train = Y_bold[train_index]
    Y_test = Y_bold[test_index]
    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)

    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )
    pipeline["multiplekernelridgecv"].cv = PredefinedSplit(run_ids[train_index])
    pipeline.fit(X_train, Y_train)

    Y_test_pred_split = pipeline.predict(X_test, split=True)
    split_scores_mask = correlation_score_split(Y_test, Y_test_pred_split)

    enc_model = pipeline["multiplekernelridgecv"]
    cv_models.append(enc_model)
    cv_scores.append(split_scores_mask)
    cv_alphas.append(enc_model.best_alphas_)


print("Saving", datetime.now())
result = {
    "masker": sub_maskers[0],
    "cv_models": cv_models,
    "cv_scores": cv_scores,
    "cv_alphas": cv_alphas,
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
