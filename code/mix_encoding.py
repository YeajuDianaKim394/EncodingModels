#!/scratch/gpfs/zzada/conda-envs/fconv/bin/python
# SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
# SBATCH --mem=8G                 # memory per cpu-core (4G is default)
# SBATCH --nodes=1                # node count
# SBATCH --ntasks=1               # total number of tasks across all nodes
# SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
# SBATCH --job-name=blenc         # create a short name for your job
# SBATCH --gres=gpu:1             # get a gpu
# SBATCH -o 'logs/%A_convblack.log'
# SBATCH --mail-type=FAIL
# SBATCH --mail-user=zzada@princeton.edu

import warnings

import black_encoding as benc
import h5py
import numpy as np
import torch
from constants import SUBS_STRANGERS
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from util import subject
from util.path import Path
from voxelwise_tutorials.delayer import Delayer

import encoding as cenc

warnings.simplefilter(action="ignore", category=FutureWarning)

TRS = 534
# the first 12 seconds is the ready screen (8 TRs), 13:20 (800 s) the actual
# story (534 TRs), and last 12 seconds (8 TRs)  done screen.


def build_black_regressors(modelname=None, layer=0):
    lexical_embs = benc.get_llm_embs(modelname=modelname, layer=layer)
    phone_rates, _ = benc.get_phoneme_features()
    word_onsets, word_rates = benc.get_transcript_features()

    X = np.hstack(
        (
            word_onsets.reshape(-1, 1),
            word_rates.reshape(-1, 1),
            phone_rates.reshape(-1, 1),
            lexical_embs,
        )
    )

    slices = {
        "task": slice(0, 3),
        "lexical": slice(3, X.shape[1]),
    }

    return X, slices


def build_conv_regressors(sub, modelname):
    spaces = {
        "task": ["prod_onset", "comp_onset"],
        "rate": ["prod_word_rate", "comp_word_rate"],
        "phone_rate": ["prod_phoneme_rate", "comp_phoneme_rate"],
        "prod_semantic": ["prod_lexical_emb"],
        "comp_semantic": ["comp_lexical_emb"],
        "in_prod": ["prod_screen"],
    }
    X, features = cenc.build_regressors(sub, modelname, spaces=spaces)

    emb_dim = features["prod_semantic"].stop - features["prod_semantic"].start
    X_conv = np.zeros((len(X), 3 + emb_dim), dtype=np.float32)
    X_conv[:, 0] = X[:, 0] + X[:, 1]  # onset
    X_conv[:, 1] = X[:, 2] + X[:, 3]  # word rate
    X_conv[:, 2] = X[:, 4] + X[:, 5]  # phoneme rate
    X_conv[:, 3:] = X[:, features["prod_semantic"]] + X[:, features["comp_semantic"]]

    return X_conv, X[:, features["in_prod"]]


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    verbose: int,
    n_jobs: int,
):
    """Build the pipeline"""

    # Make kernelizer
    kernelizers_tuples = [
        (name, Kernelizer(kernel="linear"), slice_)
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
    layer = args.layer

    args.use_cache = True
    args.cache_desc = "trialmot6"

    X_story, features = build_black_regressors(modelname=args.model, layer=layer)
    feature_names = list(features.keys())
    slices = list(features.values())

    pipeline = build_model(
        feature_names,
        slices,
        args.alphas,
        args.verbose,
        args.jobs,
    )

    # NOTE - this will apply standard scalar separately to each dataset
    delays = [2, 3, 4, 5]
    black_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False), Delayer(delays=delays)
    )
    X_story = black_pipeline.fit_transform(X_story)

    conv_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        cenc.SplitDelayer(delays=delays),
    )

    # not all subjects have this..
    subs = list(SUBS_STRANGERS)
    subs.remove(11)
    subs.remove(111)
    subs.remove(12)

    for sub in tqdm(subs):
        X_conv, prod_mask = build_conv_regressors(
            sub, f"model-{args.model}_layer-{layer}"
        )
        delayer = cenc.SplitDelayer(delays=[2, 3, 4, 5])
        prod_mask = delayer.fit_transform(prod_mask).any(-1)

        Y_story = benc.get_bold(sub)
        Y_conv = subject.get_bold(
            sub, use_cache=args.use_cache, cache_desc=args.cache_desc
        )

        X_conv = conv_pipeline.fit_transform(X_conv)

        # Train on story and test it on conv
        # pipeline.fit(X_story, Y_story)

        # Train on conv and test it on story
        pipeline.fit(X_conv, Y_conv)

        Y_preds = pipeline.predict(X_story, split=True)
        scores_story = correlation_score_split(Y_story, Y_preds)

        # Test on conversation
        Y_preds = pipeline.predict(X_conv, split=True)
        scores_conv = correlation_score_split(Y_conv, Y_preds)
        scores_prod = correlation_score_split(Y_conv[prod_mask], Y_preds[:, prod_mask])
        scores_comp = correlation_score_split(
            Y_conv[~prod_mask], Y_preds[:, ~prod_mask]
        )

        results = {
            "scores_story": scores_story.numpy(force=True),
            "scores_conv": scores_conv.numpy(force=True),
            "scores_prod": scores_prod.numpy(force=True),
            "scores_comp": scores_comp.numpy(force=True),
        }

        # save
        pklpath = Path(
            # root="encoding/black_to_conv",
            root="encoding/conv_to_black",
            sub=f"{sub:03d}",
            datatype=f"model-{args.model}_layer-{layer}",
            ext=".hdf5",
        )
        pklpath.mkdirs()
        with h5py.File(pklpath, "w") as f:
            for key, value in results.items():
                f.create_dataset(name=key, data=value)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="opt-7b")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.alphas = np.logspace(0, 19, 20)

    if args.cuda > 0:
        if torch.cuda.is_available():
            set_backend("torch_cuda")
        else:
            print("[WARN] cuda not available")

    main(args)
