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
from scipy.ndimage import binary_dilation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.path import Path
from util.subject import get_bold, get_button_presses, get_transcript


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


def get_regressors(subject: int, modelname: str):
    conv = str(subject + 100 if subject < 100 else subject)
    dfemb = get_transcript(conv=conv, modelname=modelname)
    dfphone = get_transcript(conv=conv, modelname="articulatory")

    # Build regressors per TR
    audio_emb = []
    # conf_emb = []
    regressors = defaultdict(list)

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

        audpath = Path(root="features_wx", datatype="spectrogram", ext=".npy")
        audpath.update(conv=conv, run=run, trial=trial)
        filename = glob(audpath.starstr(["conv", "datatype"]))[0]
        audio_emb.append(np.load(filename))
        print(audio_emb[-1].shape, audpath)

        # confpath = Path(root="features", datatype="motion", ext=".npy")
        # modtrial = ((trial - 1) % 4) + 1
        # confpath.update(sub=f"{subject:03d}", run=run, trial=modtrial)
        # conf_emb.append(np.load(confpath))

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

        regressors["prod_onset"].append(in_prod)
        regressors["prod_word_rate"].append(prod_wr)
        regressors["prod_phoneme_rate"].append(prod_pr)
        regressors["prod_phoneme_emb"].append(prod_phoemb)
        regressors["prod_lexical_emb"].append(prod_lexemb)

        regressors["comp_onset"].append(in_comp)
        regressors["comp_word_rate"].append(comp_wr)
        regressors["comp_phoneme_rate"].append(comp_pr)
        regressors["comp_phoneme_emb"].append(comp_phoemb)
        regressors["comp_lexical_emb"].append(comp_lexemb)

    # get additional nuisance variables
    presses, switches = get_button_presses(subject)
    presses = binary_dilation(presses)
    regressors["prod_button_press"] = presses.reshape(-1, 1)
    regressors["prod_screen"] = switches.reshape(-1, 1)
    regressors["comp_screen"] = 1 - switches.reshape(-1, 1)

    # split spectral embeddings
    breakpoint()
    pmask = switches.flatten().astype(bool)
    audio_emb = np.vstack(audio_emb)
    prod_audemb = np.zeros_like(audio_emb)
    comp_audemb = np.zeros_like(audio_emb)
    prod_audemb[pmask] = audio_emb[pmask]
    comp_audemb[~pmask] = audio_emb[~pmask]
    regressors["prod_spectral_emb"] = prod_audemb
    regressors["comp_spectral_emb"] = comp_audemb

    # # split motion embeddings
    # conf_emb = np.vstack(conf_emb)
    # prod_confemb = np.zeros_like(conf_emb)
    # comp_confemb = np.zeros_like(conf_emb)
    # prod_confemb[pmask] = conf_emb[pmask]
    # comp_confemb[~pmask] = conf_emb[~pmask]
    # regressors["prod_motion_emb"] = prod_confemb
    # regressors["comp_motion_emb"] = comp_confemb

    return regressors


def build_regressors(subject: int, modelname: str, spaces: dict = None):
    regressors = get_regressors(subject, modelname)

    if spaces is None:
        spaces = {
            "task": [
                "prod_button_press",
                "prod_screen",
                "prod_onset",
                "prod_word_rate",
                "prod_phoneme_rate",
                "comp_screen",
                "comp_onset",
                "comp_word_rate",
                "comp_phoneme_rate",
            ],
            # "motion": ["prod_motion_emb", "comp_motion_emb"],
            "spectral": ["prod_spectral_emb", "comp_spectral_emb"],
            "articulation": ["prod_phoneme_emb", "comp_phoneme_emb"],
            "prod_semantic": ["prod_lexical_emb"],
            "comp_semantic": ["comp_lexical_emb"],
        }

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
    modelname = args.model

    spaces = None
    if modelname == "acoustic":
        modelname = "model-gpt2-2b_layer-24"
        spaces = {
            "task": [
                "prod_button_press",
                "prod_screen",
                "prod_onset",
                "prod_word_rate",
                "prod_phoneme_rate",
                "comp_screen",
                "comp_onset",
                "comp_word_rate",
                "comp_phoneme_rate",
            ],
            "prod_spectral": ["prod_spectral_emb"],
            "comp_spectral": ["comp_spectral_emb"],
        }
    elif modelname == "articulatory":
        modelname = "model-gpt2-2b_layer-24"
        spaces = {
            "task": [
                "prod_button_press",
                "prod_screen",
                "prod_onset",
                "prod_word_rate",
                "prod_phoneme_rate",
                "comp_screen",
                "comp_onset",
                "comp_word_rate",
                "comp_phoneme_rate",
            ],
            "prod_articulatory": ["prod_phoneme_emb"],
            "comp_articulatory": ["comp_phoneme_emb"],
        }
    elif modelname == "static":
        modelname = "model-gpt2-2b_layer-0"
        spaces = {
            "task": [
                "prod_button_press",
                "prod_screen",
                "prod_onset",
                "prod_word_rate",
                "prod_phoneme_rate",
                "comp_screen",
                "comp_onset",
                "comp_word_rate",
                "comp_phoneme_rate",
            ],
            "prod_llm": ["prod_lexical_emb"],
            "comp_llm": ["comp_lexical_emb"],
        }
    elif modelname == "contextual":
        modelname = "model-gpt2-2b_layer-24"
        spaces = {
            "task": [
                "prod_button_press",
                "prod_screen",
                "prod_onset",
                "prod_word_rate",
                "prod_phoneme_rate",
                "comp_screen",
                "comp_onset",
                "comp_word_rate",
                "comp_phoneme_rate",
            ],
            "prod_llm": ["prod_lexical_emb"],
            "comp_llm": ["comp_lexical_emb"],
        }
    elif modelname == "syntactic":
        modelname = "syntactic"
        spaces = {
            "task": [
                "prod_button_press",
                "prod_screen",
                "prod_onset",
                "prod_word_rate",
                "prod_phoneme_rate",
                "comp_screen",
                "comp_onset",
                "comp_word_rate",
                "comp_phoneme_rate",
            ],
            "prod_llm": ["prod_lexical_emb"],
            "comp_llm": ["comp_lexical_emb"],
        }

    X, features = build_regressors(sub, modelname, spaces=spaces)
    X = X.astype(np.float32)
    feature_names = list(features.keys())
    slices = list(features.values())

    # remove any uninformative dimensions for syntactic only
    if modelname == "syntactic":
        missingMask = X.sum(0) > 0
        if not np.all(missingMask):
            # print("WARNING: contains features with all 0s")
            n1 = (~missingMask[slices[1]]).sum()
            n2 = (~missingMask[slices[2]]).sum()
            X = X[:, missingMask]
            slices[1] = slice(slices[1].start, slices[1].stop - n1)
            slices[2] = slice(slices[2].start - n1, slices[2].stop - n1 - n2)

    delayer = SplitDelayer(delays=[2, 3, 4, 5])
    pipeline = build_model(feature_names, slices, args.alphas, args.verbose, args.jobs)

    print(datetime.now(), "Get BOLD")
    Y_bold = get_bold(
        sub, atlas=args.atlas, use_cache=args.use_cache, cache_desc=args.cache_desc
    )

    results = defaultdict(list)
    run_ids = np.repeat(RUNS, CONV_TRS * 2)
    kfold = PredefinedSplit(run_ids)
    # kfold = KFold(n_splits=2)
    for k, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]
        print(datetime.now(), f"F{k+1}", X_train.shape, Y_train.shape)

        pipeline["multiplekernelridgecv"].cv = PredefinedSplit(run_ids[train_index])  # type: ignore
        pipeline.fit(X_train, Y_train)

        Y_preds = pipeline.predict(X_test, split=True)
        scores_split = correlation_score_split(Y_test, Y_preds)

        # Compute correlation based on masked prod/comp
        # 1 index may change below depending on features
        prod_mask = X_test[:, 1:2]
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
        results["cv_preds"].append(Y_preds.numpy(force=True))
        results["cv_prodmask"].append(prod_mask)

    # stack across folds
    result = {k: np.stack(v) for k, v in results.items()}
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--subject", type=int)
    parser.add_argument("-m", "--model", type=str, default="model-gpt2-2b_layer-24")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--atlas", type=str, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-desc", type=str, default=None)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.alphas = np.logspace(0, 19, 20)
    print(datetime.now(), "Start", args.model, args.cache_desc)

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
