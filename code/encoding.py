"""
Run an encoding model on BOLD data.

https://gallantlab.org/voxelwise_tutorials/_auto_examples/shortclips/03_plot_wordnet_model.html
https://gallantlab.org/voxelwise_tutorials/_auto_examples/shortclips/06_plot_banded_ridge_model.html
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from glob import glob

import h5py
import numpy as np
import torch
from constants import CONV_TRS, RUNS, TR
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score, correlation_score_split
from scipy.ndimage import binary_dilation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.atlas import Atlas
from util.path import Path
from util.subject import get_bold, get_button_presses, get_transcript


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
    dfphone = get_transcript(conv=conv, modelname="articulatory")

    # Build regressors per TR
    X = []
    dim_lexemb = 0
    dim_phoemb = 0
    dim_audemb = 80
    audio_emb = []

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

        audpath = Path(root="features", datatype="spectrogram", ext=".npy")
        audpath.update(conv=conv, run=run, trial=trial)
        filename = glob(audpath.starstr(["conv", "datatype"]))[0]
        audio_emb.append(np.load(filename))

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

    # get additional nuisance variables
    presses, switches = get_button_presses(subject)
    presses = binary_dilation(presses)

    pmask = switches.astype(bool)

    # split spectral embeddings
    audio_emb = np.vstack(audio_emb)
    prod_audemb = np.zeros_like(audio_emb)
    comp_audemb = np.zeros_like(audio_emb)
    prod_audemb[pmask] = audio_emb[pmask]
    comp_audemb[~pmask] = audio_emb[~pmask]

    X = np.hstack(
        (
            presses.reshape(-1, 1),
            switches.reshape(-1, 1),
            (1 - switches).reshape(-1, 1),
            X[:, :6],
            prod_audemb,
            comp_audemb,
            X[:, 6:],
        )
    )

    n_nuisance = 9
    n_phodims = dim_phoemb * 2
    n_spedims = dim_audemb * 2
    n_lexdims = dim_lexemb

    feat_dims = [0, n_nuisance, n_spedims, n_phodims, n_lexdims, n_lexdims]
    featdim_cs = np.cumsum(feat_dims)
    features = [
        "task",
        "spectral",
        "articulation",
        "production",
        "comprehension",
    ]
    slices = {}
    for i, feat in enumerate(features, 1):
        slices[feat] = slice(featdim_cs[i - 1], featdim_cs[i])

    return X, slices


def compress_regressors(X: np.ndarray, features: dict) -> (np.ndarray, dict):
    """Compress regressors across production and comprehension,

    removing this distinction."""
    raise NotImplementedError
    x_nuis = X[:, [0, 1, 2]]
    x_in = X[:, 2:4].sum(axis=1, keepdims=True)
    x_wr = X[:, 4:6].sum(axis=1, keepdims=True)
    x_pr = X[:, 6:8].sum(axis=1, keepdims=True)
    x_ph = X[:, 8:47] + X[:, 47:86]
    x_lx = X[:, features["production"]] + X[:, features["comprehension"]]
    Xnew = np.hstack((x_nuis, x_in, x_wr, x_pr, x_ph, x_lx))
    features_new = dict(
        nuisance=slice(0, 6),
        phonemes=slice(6, 45),
        lexical=slice(45, 1069),
    )
    return Xnew, features_new


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    verbose: int,
    n_jobs: int,
):
    """Build the pipeline"""

    # Set up modeling pipeline
    delayer_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        SplitDelayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )

    # Make kernelizer
    kernelizers_tuples = [
        (name, delayer_pipeline, slice_) for name, slice_ in zip(feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=n_jobs)

    params = dict(
        alphas=alphas, progress_bar=verbose, diagonalize_method="svd", n_iter=100
    )
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

    X, features = build_regressors(sub, modelname)
    # X, features = compress_regressors(X, features)
    X = X.astype(np.float32)
    feature_names = list(features.keys())
    slices = list(features.values())

    pipeline = build_model(feature_names, slices, args.alphas, args.verbose, args.jobs)

    print(datetime.now(), "Get BOLD")
    Y_bold = get_bold(sub, atlas=args.atlas, use_cache=args.use_cache)

    if args.atlas is not None:
        # atlasname = args.atlas
        atlas = Atlas.schaefer(1000)
        Y_bold = atlas.vox_to_parc(Y_bold)
        print(Y_bold.shape)

    results = defaultdict(list)
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
        prod_mask = X_test[:, 1].astype(bool)
        comp_mask = X_test[:, 2].astype(bool)
        scores_prod = correlation_score(Y_test[prod_mask], Y_preds[-2, prod_mask])
        scores_comp = correlation_score(Y_test[comp_mask], Y_preds[-1, comp_mask])

        # test generalization from left to right hemisphere

        # test production model on comprehension embeddings and time points

        enc_model = pipeline["multiplekernelridgecv"]  # type: ignore

        if args.save_weights:
            Xfit = pipeline["columnkernelizer"].get_X_fit()
            weights = enc_model.get_primal_coef(Xfit)

            weights_prod = weights[-2]
            weights_prod_delay = weights_prod.reshape(-1, 4, weights_prod.shape[-1])
            weights_prod = weights_prod_delay.mean(1)

            weights_comp = weights[-1]
            weights_comp_delay = weights_comp.reshape(-1, 4, weights_comp.shape[-1])
            weights_comp = weights_comp_delay.mean(1)

            results["cv_weights_prod"].append(weights_prod)
            results["cv_weights_comp"].append(weights_comp)

        results["cv_scores"].append(scores_split.numpy(force=True))
        results["cv_scores_prod"].append(scores_prod.numpy(force=True))
        results["cv_scores_comp"].append(scores_comp.numpy(force=True))
        results["cv_alphas"].append(enc_model.best_alphas_.numpy(force=True))
        results["cv_preds"].append(Y_preds.numpy(force=True))

    # stack across folds
    result = {k: np.stack(v) for k, v in results.items()}
    result["in_prod"] = X[:, 1].astype(bool)
    result["in_comp"] = X[:, 2].astype(bool)

    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--subject", type=int)
    parser.add_argument(
        "-m", "--model", type=str, default="model-gpt2-medium_layer-0.75"
    )
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--atlas", type=str, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.alphas = np.logspace(0, 19, 20)
    print(datetime.now(), "Start")

    if args.cuda > 0:
        if torch.cuda.is_available():
            set_backend("torch_cuda")
        else:
            print("[WARN] cuda not available")

    result = main(args)

    print(datetime.now(), "Saving")
    desc = ""
    if args.atlas is not None:
        desc += args.atlas
    pklpath = Path(
        root=f"encoding{args.suffix}",
        sub=f"{args.subject:03d}",
        datatype=args.model,
        desc=desc,
        ext=".hdf5",
    )
    pklpath.mkdirs()
    with h5py.File(pklpath, "w") as f:
        for key, value in result.items():
            f.create_dataset(name=key, data=value)
