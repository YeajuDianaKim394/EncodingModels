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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.path import Path
from util.subject import get_bold, get_button_presses, get_transcript, word_to_phones


class Caster(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return torch.Tensor(X)


class SplitDelayer(BaseEstimator, TransformerMixin):
    def __init__(self, delays=None):
        self.delays = delays

    def fit(self, X, y=None):
        # print("Delayer fit", X.shape)
        X = self._validate_data(X, dtype="numeric")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        # print("Delayer transform", X.shape)
        # split X, apply delayer to each, then restitch
        n_samples = X.shape[0]
        if n_samples % 120 != 0:
            raise ValueError(f"n_samples is not divisible by 120, {n_samples}")
        n_splits = n_samples // 120  # number of trials
        return np.vstack([self._delay(x) for x in np.split(X, n_splits, axis=0)])

    def _delay(self, X):
        # Borrowed from himalaya
        if self.delays is None:
            return X

        n_samples, n_features = X.shape

        X_delayed = np.zeros((n_samples, n_features * len(self.delays)), dtype=X.dtype)
        for idx, delay in enumerate(self.delays):
            beg, end = idx * n_features, (idx + 1) * n_features
            if delay == 0:
                X_delayed[:, beg:end] = X
            elif delay > 0:
                X_delayed[delay:, beg:end] = X[:-delay]
            elif delay < 0:
                X_delayed[: -abs(delay), beg:end] = X[abs(delay) :]

        return X_delayed


def build_regressors(subject: int, modelname: str):
    conv = str(subject + 100 if subject < 100 else subject)
    dfemb = get_transcript(conv=conv, modelname=modelname)
    dfphone = word_to_phones(dfemb)

    # Build regressors per TR
    X = []
    dim_lexemb = 0
    dim_phoemb = 0

    # For each trial, create a feature for each TR
    for (run, trial), subdf in dfemb.groupby(["run", "trial"]):
        dim_lexemb = len(subdf.iloc[0].embedding)
        prod_wr = np.zeros((CONV_TRS, 1))
        comp_wr = np.zeros((CONV_TRS, 1))
        in_prod = np.zeros((CONV_TRS, 1), dtype=bool)
        in_comp = np.zeros((CONV_TRS, 1), dtype=bool)
        prod_lexemb = np.zeros((CONV_TRS, dim_lexemb))
        comp_lexemb = np.zeros((CONV_TRS, dim_lexemb))
        is_prod = subdf.speaker == subject
        is_comp = subdf.speaker != subject

        subdf_phones = dfphone[(dfphone.run == run) & (dfphone.trial == trial)]
        prod_pr = np.zeros((CONV_TRS, 1))
        comp_pr = np.zeros((CONV_TRS, 1))
        dim_phoemb = len(subdf_phones.iloc[0].embedding)
        prod_phoemb = np.zeros((CONV_TRS, dim_phoemb))
        comp_phoemb = np.zeros((CONV_TRS, dim_phoemb))
        is_prod_pho = subdf_phones.speaker == subject
        is_comp_pho = subdf_phones.speaker != subject

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
                prod_wr[t] = prod_mask.sum()
                tr_embedding = np.vstack(subdf[prod_mask].embedding).mean(  # type: ignore
                    axis=0, keepdims=True
                )
                prod_lexemb[t] = tr_embedding

            comp_mask = mask & is_comp
            if comp_mask.any():
                in_comp[t] = True
                comp_wr[t] = comp_mask.sum()
                tr_embedding = np.vstack(subdf[comp_mask].embedding).mean(  # type: ignore
                    axis=0, keepdims=True
                )
                comp_lexemb[t] = tr_embedding

            # repeat for phonemes
            pho_mask = (subdf_phones.start <= end_s) & (subdf_phones.start > start_s)

            prod_mask = pho_mask & is_prod_pho
            if prod_mask.any():
                prod_pr[t] = np.sum(subdf_phones[prod_mask].embedding.sum())
                prod_phoemb[t] = np.vstack(subdf_phones[prod_mask].embedding).any(
                    axis=0, keepdims=True
                )

            comp_mask = pho_mask & is_comp_pho
            if comp_mask.any():
                comp_pr[t] = np.sum(subdf_phones[comp_mask].embedding.sum())
                comp_phoemb[t] = np.vstack(subdf_phones[comp_mask].embedding).any(
                    axis=0, keepdims=True
                )

        # Convolve prod/comp indicators
        x_trial = np.hstack(
            (
                in_prod,
                in_comp,
                prod_wr,
                comp_wr,
                prod_pr,
                comp_pr,
                prod_phoemb,
                comp_phoemb,
                prod_lexemb,
                comp_lexemb,
            )
        )
        X.append(x_trial)

    X = np.vstack(X)

    presses, switches = get_button_presses(subject)
    X = np.hstack((presses.reshape(-1, 1), switches.reshape(-1, 1), X))

    n_nuisance = 4
    n_rates = 4
    n_phodims = dim_phoemb * 2
    n_lexdims = dim_lexemb

    feat_dims = [0, n_nuisance, n_rates, n_phodims, n_lexdims, n_lexdims]
    featdim_cs = np.cumsum(feat_dims)
    features = ["nuisance", "wp_rate", "phonemes", "production", "comprehension"]

    slices = {}
    for i, feat in enumerate(features, 1):
        slices[feat] = slice(featdim_cs[i - 1], featdim_cs[i])

    return X, slices


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    verbose: int,
    n_jobs: int,
):
    # Set up modeling pipeline
    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        SplitDelayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )

    # Make kernelizer
    kernelizers_tuples = [
        (name, preprocess_pipeline, slice_)
        for name, slice_ in zip(feature_names, slices)
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
    X = X.astype(np.float32)
    feature_names = list(features.keys())
    slices = list(features.values())

    pipeline = build_model(feature_names, slices, args.alphas, args.verbose, args.jobs)

    print(datetime.now(), "Get BOLD")
    # Y_bold, fw_displacement = get_bold(
    #     sub, space="fsaverage6", return_cofounds=["framewise_displacement"]
    # )
    Y_bold = get_bold(sub, space="fsaverage6")

    delayer = SplitDelayer(delays=[2, 3, 4, 5])

    run_ids = np.repeat(RUNS, CONV_TRS * 2)
    kfold = PredefinedSplit(run_ids)
    for k, (train_index, test_index) in enumerate(kfold.split()):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]
        print(
            datetime.now(),
            "Fold",
            k + 1,
            X_train.shape,
            Y_train.shape,
            X.dtype,
            Y_bold.dtype,
        )

        pipeline["multiplekernelridgecv"].cv = PredefinedSplit(run_ids[train_index])  # type: ignore

        pipeline.fit(X_train, Y_train)

        Y_preds = pipeline.predict(X_test, split=True)
        scores_split = correlation_score_split(Y_test, Y_preds)

        # Compute correlation based on masked prod/comp
        prod_mask = delayer.fit_transform(X_test[:, 2:3]).any(-1)
        comp_mask = delayer.fit_transform(X_test[:, 3:4]).any(-1)
        scores_prod = correlation_score(Y_test[prod_mask], Y_preds[-2, prod_mask])
        scores_comp = correlation_score(Y_test[comp_mask], Y_preds[-1, comp_mask])

        enc_model = pipeline["multiplekernelridgecv"]  # type: ignore
        cv_models.append(enc_model)
        cv_scores.append(scores_split.numpy(force=True))
        cv_scores_prod.append(scores_prod.numpy(force=True))
        cv_scores_comp.append(scores_comp.numpy(force=True))
        cv_alphas.append(enc_model.best_alphas_.numpy(force=True))
        cv_preds.append(Y_preds.numpy(force=True))

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
    parser = ArgumentParser()
    parser.add_argument("-s", "--subject", type=int)
    parser.add_argument(
        "-m", "--model", type=str, default="model-gpt2-medium_layer-0.75"
    )
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.alphas = np.logspace(-1, 8, 10)
    print(datetime.now(), "Start")

    if args.cuda > 0:
        if torch.cuda.is_available():
            set_backend("torch_cuda")
        else:
            print("[WARN] cuda not available")

    result = main(args)

    print(datetime.now(), "Saving")
    pklpath = Path(
        root="encoding",
        sub=f"{args.subject:03d}",
        datatype=args.model,
        ext=".pkl",
    )
    pklpath.mkdirs()
    with open(pklpath.fpath, "wb") as f:
        pickle.dump(result, f)
