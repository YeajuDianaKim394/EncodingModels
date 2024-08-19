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
from himalaya.scoring import correlation_score_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.atlas import Atlas
from util.path import Path
from util.subject import (
    get_bold,
    get_conv,
    get_timinglog_boxcars,
    get_transcript_features,
)

TASK_REGRESSORS = [
    "prod_screen",
    "prod_word_onset",
    "prod_word_rate",
    "prod_phoneme_rate",
    "comp_screen",
    "comp_word_onset",
    "comp_word_rate",
    "comp_phoneme_rate",
]

SPACES = {
    "acoustic": {
        "task": TASK_REGRESSORS,
        "prod_spectral": ["prod_spectral_emb"],
        "comp_spectral": ["comp_spectral_emb"],
    },
    "phonemic": {
        "task": TASK_REGRESSORS,
        "prod_articulatory": ["prod_phoneme_emb"],
        "comp_articulatory": ["comp_phoneme_emb"],
    },
    "articulatory": {
        "task": TASK_REGRESSORS,
        "prod_articulatory": ["prod_phoneme_emb"],
        "comp_articulatory": ["comp_phoneme_emb"],
    },
    "syntactic": {
        "task": TASK_REGRESSORS,
        "prod_llm": ["prod_lexical_emb"],
        "comp_llm": ["comp_lexical_emb"],
    },
    "static": {
        "task": TASK_REGRESSORS,
        "prod_llm": ["prod_lexical_emb"],
        "comp_llm": ["comp_lexical_emb"],
    },
    "contextual": {
        "task": TASK_REGRESSORS,
        "prod_llm": ["prod_lexical_emb"],
        "comp_llm": ["comp_lexical_emb"],
    },
    "joint": {
        "task": TASK_REGRESSORS,
        "spectral": ["prod_spectral_emb", "comp_spectral_emb"],
        "articulation": ["prod_phoneme_emb", "comp_phoneme_emb"],
        "prod_semantic": ["prod_lexical_emb"],
        "comp_semantic": ["comp_lexical_emb"],
    },
    "joint_syntactic": {
        "task": TASK_REGRESSORS,
        "spectral": ["prod_spectral_emb", "comp_spectral_emb"],
        "articulation": ["prod_phoneme_emb", "comp_phoneme_emb"],
        "prod_llm": ["prod_lexical_emb"],
        "comp_llm": ["comp_lexical_emb"],
    },
    "joint_nosplit": {
        "task": [
            "prod_screen",
            "comp_screen",
            "word_onset",
            "word_rate",
            "phoneme_rate",
        ],
        "spectral": ["spectral_emb"],
        "articulation": ["phoneme_emb"],
        "semantic": ["lexical_emb"],
    },
}


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


def get_regressors(sub_id: int, modelname: str, split: bool = True):
    conv = get_conv(sub_id)
    dfemb = get_transcript_features(sub_id, modelname=modelname)
    dfphone = get_transcript_features(sub_id, modelname="articulatory")
    # dfphone = get_transcript_features(sub_id, modelname="phonemic")

    regressors = defaultdict(list)

    # For each trial, create a feature for each TR
    for (run, trial), subdf in dfemb.groupby(["run", "trial"]):
        word_rate = np.zeros((CONV_TRS, 1))
        phoneme_rate = np.zeros((CONV_TRS, 1))
        word_onset = np.zeros((CONV_TRS, 1))
        dim_lexemb = len(subdf.iloc[0].embedding)
        lexical_emb = np.zeros((CONV_TRS, dim_lexemb))

        subdf_phones = dfphone[(dfphone.run == run) & (dfphone.trial == trial)]
        dim_phoemb = len(subdf_phones.iloc[0].embedding)
        phone_emb = np.zeros((CONV_TRS, dim_phoemb))

        # Go through one TR at a time and find words that fall within this TR
        # average their embeddings, get num of words, etc
        for t in range(CONV_TRS):
            start_s = t * TR
            end_s = start_s + TR
            mask = (subdf.start <= end_s) & (subdf.start > start_s)
            if mask.any():
                word_onset[t] = 1
                word_rate[t] = mask.sum()
                lexical_emb[t] = np.vstack(subdf[mask].embedding).mean(
                    axis=0, keepdims=True
                )

            mask = (subdf_phones.start <= end_s) & (subdf_phones.start > start_s)
            if mask.any():
                phoneme_rate[t] = mask.sum()
                phone_emb[t] = np.vstack(subdf_phones[mask].embedding).mean(
                    axis=0, keepdims=True
                )

        regressors["word_onset"].append(word_onset)
        regressors["word_rate"].append(word_rate)
        regressors["phoneme_rate"].append(phoneme_rate)
        regressors["phoneme_emb"].append(phone_emb)
        regressors["lexical_emb"].append(lexical_emb)

        audpath = Path(root="data/stimuli", datatype="spectrogram", ext=".npy")
        audpath.update(conv=conv, run=run, trial=trial)
        filename = glob(audpath.starstr(["conv", "datatype"]))[0]
        audio_emb = np.load(filename)
        regressors["spectral_emb"].append(audio_emb)
        # print(audio_emb[-1].shape, audpath)

    # remove any uninformative dimensions for syntactic only
    if modelname == "syntactic":
        values = np.concatenate(regressors["lexical_emb"])
        missingMask = values.sum(0) > 0
        if not np.all(missingMask):
            print("[WARNING] contains features with all 0s", missingMask.sum())
            values = values[:, missingMask]
        regressors["lexical_emb"] = values

    # add additional nuisance variables
    prod_boxcar, _, _ = get_timinglog_boxcars(sub_id)
    prod_mask = prod_boxcar.astype(bool)

    if split:
        split_regressors = defaultdict(list)
        for regressor, values in regressors.items():
            if isinstance(values, list) and len(values) > 1:
                values = np.concatenate(values)
            else:
                values = np.asarray(values)

            prod_values = np.zeros_like(values)
            prod_values[prod_mask] = values[prod_mask]
            split_regressors[f"prod_{regressor}"] = prod_values

            comp_values = np.zeros_like(values)
            comp_values[~prod_mask] = values[~prod_mask]
            split_regressors[f"comp_{regressor}"] = comp_values

        regressors = split_regressors

    regressors["prod_screen"] = prod_boxcar.reshape(-1, 1)
    regressors["comp_screen"] = 1 - prod_boxcar.reshape(-1, 1)

    return regressors


def build_regressors(subject: int, modelname: str, spaces: dict = None):
    regressors = get_regressors(subject, modelname, split=True)  # NOTE split

    X = []
    start = 0
    slices = {}
    for feature_space, items in spaces.items():
        x_features = []
        for item in items:
            values = regressors[item]
            if isinstance(values, list) and len(values) > 1:
                values = np.concatenate(values)
            else:
                values = np.asarray(values)
            x_features.append(values)
            # print(subject, modelname, feature_space, type(values), values.shape)
        x_features = np.hstack(x_features)
        X.append(x_features)
        slices[feature_space] = slice(start, start + x_features.shape[1])
        start += x_features.shape[1]

    X = np.hstack(X)
    return X, slices


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    verbose: int,
    n_jobs: int,
):
    """Build the pipeline"""

    # Set up modeling pipeline
    default_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        SplitDelayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )
    task_pipeline = make_pipeline(
        SplitDelayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )
    motion_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Kernelizer(kernel="linear"),
    )
    space2pipe = {"task": task_pipeline, "motion": motion_pipeline}

    # Make kernelizer
    kernelizers_tuples = [
        (name, space2pipe.get(name, default_pipeline), slice_)
        for name, slice_ in zip(feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=n_jobs)

    params = dict(
        alphas=alphas, progress_bar=verbose, diagonalize_method="svd", n_iter=100
    )
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )

    return pipeline


def main(args):
    sub = args.subject
    space = args.model
    modelname = args.lang_model

    spaces = SPACES[space]
    X, features = build_regressors(sub, modelname, spaces=spaces)
    X = X.astype(np.float32)
    feature_names = list(features.keys())
    slices = list(features.values())

    delayer = SplitDelayer(delays=[2, 3, 4, 5])
    pipeline = build_model(feature_names, slices, args.alphas, args.verbose, args.jobs)

    Y_bold = get_bold(sub, cache=args.cache)
    if args.atlas is not None:
        atlas = Atlas.schaefer(100)
        Y_bold = atlas.vox_to_parc(Y_bold)
        print(Y_bold.shape)

    results = defaultdict(list)
    run_ids = np.repeat(RUNS, CONV_TRS * 2)
    kfold = PredefinedSplit(run_ids)
    # kfold = KFold(n_splits=2)
    for k, (train_index, test_index) in enumerate(kfold.split(X)):
        # for k, (test_index, train_index) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]
        print(
            datetime.now(),
            f"F{k+1}",
            X_train.shape,
            Y_train.shape,
            X_test.shape,
            Y_test.shape,
        )

        pipeline["multiplekernelridgecv"].cv = PredefinedSplit(run_ids[train_index])  # type: ignore
        pipeline.fit(X_train, Y_train)

        Y_preds = pipeline.predict(X_test, split=True)
        scores_split = correlation_score_split(Y_test, Y_preds)

        # Compute correlation based on masked prod/comp
        # 1 index may change below depending on features
        prod_mask = X_test[:, 0:1]
        prod_mask = delayer.fit_transform(prod_mask).any(-1)
        comp_mask = np.logical_not(prod_mask)

        scores_prod = correlation_score_split(
            Y_test[prod_mask], Y_preds[:, prod_mask, :]
        )
        scores_comp = correlation_score_split(
            Y_test[comp_mask], Y_preds[:, comp_mask, :]
        )

        enc_model = pipeline["multiplekernelridgecv"]  # type: ignore
        if args.save_weights:
            Xfit = pipeline["columnkernelizer"].get_X_fit()
            weights = enc_model.get_primal_coef(Xfit)

            weights_prod = weights[-2]
            weights_prod_delay = weights_prod.reshape(-1, 4, weights_prod.shape[-1])
            weights_prod = weights_prod_delay.mean(1)
            # weights_prod = torch.linalg.vector_norm(weights_prod_delay, dim=0)

            weights_comp = weights[-1]
            weights_comp_delay = weights_comp.reshape(-1, 4, weights_comp.shape[-1])
            weights_comp = weights_comp_delay.mean(1)
            # weights_comp = torch.linalg.vector_norm(weights_comp_delay, dim=0)

            results["cv_weights_prod"].append(weights_prod)
            results["cv_weights_comp"].append(weights_comp)

        results["cv_scores"].append(scores_split.numpy(force=True))
        results["cv_scores_prod"].append(scores_prod.numpy(force=True))
        results["cv_scores_comp"].append(scores_comp.numpy(force=True))
        results["cv_alphas"].append(enc_model.best_alphas_.numpy(force=True))
        results["cv_prodmask"].append(prod_mask)

        if args.save_preds:
            results["cv_preds"].append(Y_preds.numpy(force=True))

    # stack across folds
    result = {k: np.stack(v) for k, v in results.items()}
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True)
    parser.add_argument("-m", "--model", type=str, default="joint")
    parser.add_argument(
        "-lm", "--lang-model", type=str, default="model-gpt2-2b_layer-24"
    )
    parser.add_argument("--cache", type=str, default="trialmot9")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--atlas", type=str, default=None)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.alphas = np.logspace(0, 19, 20)
    print(datetime.now(), "Start", args.model, args.cache)

    if args.cuda > 0:
        if torch.cuda.is_available():
            set_backend("torch_cuda")
        else:
            print("[WARN] cuda not available")

    result = main(args)

    print(datetime.now(), "Saving")
    desc = None
    # desc = "folds-2"
    if args.atlas is not None:
        desc = args.atlas
    pklpath = Path(
        root=f"results/encoding{args.suffix}",
        sub=f"{args.subject:03d}",
        datatype=args.model,
        desc=desc,
        ext=".hdf5",
    )
    pklpath.mkdirs()
    with h5py.File(pklpath, "w") as f:
        for key, value in result.items():
            f.create_dataset(name=key, data=value)
