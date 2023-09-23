"""
Run 
"""
import pickle
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from constants import CONV_TRS, RUNS, TR
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score, correlation_score_split

# from nilearn.glm.first_level import glover_hrf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.path import Path
from util.subject import get_bold, get_transcript
from voxelwise_tutorials.delayer import Delayer


class Caster(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return torch.Tensor(X)


def build_regressors(subject: int, modelname: str):
    conv = str(subject + 100 if subject < 100 else subject)
    dfemb = get_transcript(conv=conv, modelname=modelname)

    # Build regressors per TR
    X = []
    embdim = 0
    n_nuissance = 3
    # hrf = glover_hrf(TR, oversampling=1, time_length=32)

    # For each trial, create a feature for each TR
    for (_, _), subdf in dfemb.groupby(["run", "trial"]):
        embdim = len(subdf.iloc[0].embedding)
        n_words = np.zeros((CONV_TRS, 1))
        in_prod = np.zeros((CONV_TRS, 1), dtype=bool)
        in_comp = np.zeros((CONV_TRS, 1), dtype=bool)

        embeddings = np.zeros((CONV_TRS, embdim))
        prod_embeddings = np.zeros_like(embeddings)
        comp_embeddings = np.zeros_like(embeddings)

        is_prod = subdf.speaker == subject
        is_comp = subdf.speaker != subject

        # Go through one TR at a time and find words that fall within this TR
        # average their embeddings, get num of words, etc
        for t in range(CONV_TRS):
            start_s = t * TR
            end_s = start_s + TR
            mask = (subdf.start <= end_s) & (subdf.start > start_s)
            if not mask.any():
                continue

            prod_mask = mask & is_prod
            if prod_mask.any():
                in_prod[t] = True
                n_words[t] = prod_mask.sum()
                tr_embedding = np.vstack(subdf[prod_mask].embedding).mean(  # type: ignore
                    axis=0, keepdims=True
                )
                prod_embeddings[t] = tr_embedding

            comp_mask = mask & is_comp
            if comp_mask.any():
                in_comp[t] = True
                n_words[t] = comp_mask.sum()
                tr_embedding = np.vstack(subdf[comp_mask].embedding).mean(  # type: ignore
                    axis=0, keepdims=True
                )
                comp_embeddings[t] = tr_embedding

        # Convolve prod/comp indicators
        # prod_id = np.convolve(in_prod, hrf)[:CONV_TRS].reshape(-1, 1)
        # comp_id = np.convolve(in_comp, hrf)[:CONV_TRS].reshape(-1, 1)
        # n_words = np.convolve(nwords, hrf)[:CONV_TRS].reshape(-1, 1)
        x_trial = np.hstack(
            (in_prod, in_comp, n_words, prod_embeddings, comp_embeddings)
        )
        X.append(x_trial)

    X = np.vstack(X)

    slices = {
        "nuissance": slice(0, n_nuissance),  # nuissance variables
        "producton": slice(n_nuissance, embdim + n_nuissance),  # production
        "comprehension": slice(embdim + n_nuissance, X.shape[1]),  # comprehension
    }
    return X, slices


def plotboxcar(subA, subB):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    axes[0].plot(subA, color="blue")
    axes[1].plot(subB, color="red")
    axes[1].set_xlabel("time (s)")
    return fig


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    verbose: int,
    n_jobs: int,
):
    # Set up modeling pipeline
    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Delayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )

    # Set up slices of design matrix into bands
    pipelines = (pipeline, pipeline, pipeline)

    # Make kernelizer
    kernelizers_tuples = [
        (name, pipe_, slice_)
        for name, pipe_, slice_ in zip(feature_names, pipelines, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=n_jobs)

    params = dict(alphas=alphas, progress_bar=verbose, diagonalize_method="svd")
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
    pipeline = make_pipeline(
        column_kernelizer,
        Caster(),
        mkr_model,
    )

    return pipeline


def main(args):
    sub = args.subject
    modelname = args.model

    # Run encoding
    cv_scores = []
    cv_scores_prod = []
    cv_scores_comp = []
    cv_models = []
    cv_alphas = []
    cv_preds = []

    X, features = build_regressors(sub, modelname)
    feature_names = list(features.keys())
    slices = list(features.values())

    pipeline = build_model(feature_names, slices, args.alphas, args.verbose, args.jobs)

    print("Get bold", datetime.now())
    Y_bold = get_bold(sub, space="fsaverage6")

    X = X.astype(np.float32)
    # X, Y_bold = torch.Tensor(X), torch.Tensor(Y_bold)  # converts to float32 for backend
    print("shapes:", X.shape, Y_bold.shape, X.dtype, Y_bold.dtype)

    run_ids = np.repeat(RUNS, CONV_TRS * 2)
    kfold = PredefinedSplit(run_ids)
    for k, (train_index, test_index) in enumerate(kfold.split()):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]
        print("Fold", k + 1, datetime.now(), X_train.shape, Y_train.shape)

        pipeline["multiplekernelridgecv"].cv = PredefinedSplit(run_ids[train_index])  # type: ignore

        try:
            pipeline.fit(X_train, Y_train)
        except RuntimeError as e:
            print(k, e)
            continue

        Y_preds = pipeline.predict(X_test, split=True)
        scores_split = correlation_score_split(Y_test, Y_preds)

        # Compute correlation based on masked prod/comp
        prod_mask = X_test[:, 0].astype(bool)
        comp_mask = X_test[:, 1].astype(bool)
        scores_prod = correlation_score(Y_test[prod_mask], Y_preds[1, prod_mask])
        scores_comp = correlation_score(Y_test[comp_mask], Y_preds[2, comp_mask])

        enc_model = pipeline["multiplekernelridgecv"]  # type: ignore
        cv_models.append(enc_model)
        cv_scores.append(scores_split.numpy(force=True))
        cv_scores_prod.append(scores_prod.numpy(force=True))
        cv_scores_comp.append(scores_comp.numpy(force=True))
        cv_alphas.append(enc_model.best_alphas_.numpy(force=True))
        cv_preds.append(Y_preds.numpy(force=True))

    print("Saving", datetime.now())
    result = {
        "cv_scores": cv_scores,
        "cv_scores_prod": cv_scores_prod,
        "cv_scores_comp": cv_scores_comp,
        "cv_alphas": cv_alphas,
        "cv_preds": cv_preds,
        "in_prod": X[:, 0].astype(bool),
        "in_comp": X[:, 1].astype(bool),
        "cv_models": cv_models,
    }

    return result


if __name__ == "__main__":
    print("Start", datetime.now())
    parser = ArgumentParser()
    parser.add_argument("-s", "--subject", type=int)
    parser.add_argument("-m", "--model", type=str, default="gpt2")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.alphas = np.logspace(-1, 8, 10)

    backend = set_backend("torch_cuda")

    result = main(args)

    pklpath = Path(
        root="encoding",
        sub=f"{args.subject:03d}",
        datatype=args.model,
        ext=".pkl",
    )
    pklpath.mkdirs()
    with open(pklpath.fpath, "wb") as f:
        pickle.dump(result, f)
