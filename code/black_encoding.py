"""Story encoding"""
import json
from collections import defaultdict
from io import StringIO
from os import makedirs

import h5py
import numpy as np
import pandas as pd
import torch
from constants import (
    CONFOUND_REGRESSORS,
    MOTION_CONFOUNDS,
    PUNCTUATION,
    SUBS_STRANGERS,
    TR,
)
from embeddings import HFMODELS
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from util import subject
from util.path import Path
from voxelwise_tutorials.delayer import Delayer

TRS = 534
# the first 12 seconds is the ready screen (8 TRs), 13:20 (800 s) the actual
# story (534 TRs), and last 12 seconds (8 TRs)  done screen.


dfp = pd.read_csv(
    StringIO(
        """phoneme,kind,manner,place,phonation,height,position,height2,position2
B,consonant,plosive,bilabial,voiced,,,,
CH,consonant,affricate,postalveolar,unvoiced,,,,
D,consonant,plosive,alveolar,voiced,,,,
DH,consonant,fricative,dental,voiced,,,,
F,consonant,fricative,labiodental,unvoiced,,,,
G,consonant,plosive,velar,voiced,,,,
HH,consonant,fricative,glottal,unvoiced,,,,
JH,consonant,affricate,postalveolar,voiced,,,,
K,consonant,plosive,velar,unvoiced,,,,
L,consonant,lateral,alveolar,voiced,,,,
M,consonant,nasal,bilabial,voiced,,,,
N,consonant,nasal,alveolar,voiced,,,,
NG,consonant,nasal,velar,voiced,,,,
P,consonant,plosive,bilabial,unvoiced,,,,
R,consonant,approximant,alveolar,voiced,,,,
S,consonant,fricative,alveolar,unvoiced,,,,
SH,consonant,fricative,postalveolar,unvoiced,,,,
T,consonant,plosive,alveolar,unvoiced,,,,
TH,consonant,fricative,dental,unvoiced,,,,
V,consonant,fricative,labiodental,voiced,,,,
W,consonant,approximant,velar,voiced,,,,
Y,consonant,approximant,palatal,voiced,,,,
Z,consonant,fricative,alveolar,voiced,,,,
ZH,consonant,fricative,postalveolar,voiced,,,,
AA,vowel,,,,low,back,,
AE,vowel,,,,low,front,,
AH,vowel,,,,mid,central,,
AO,vowel,,,,mid,back,,
AW,vowel,,,,low,central,mid,back
AY,vowel,,,,low,central,mid,front
EH,vowel,,,,mid,front,,
ER,vowel,,,,mid,central,,
EY,vowel,,,,mid,front,,
IH,vowel,,,,mid,front,,
IY,vowel,,,,high,front,,
OW,vowel,,,,mid,back,,
OY,vowel,,,,mid,back,high,front
UH,vowel,,,,high,back,,
UW,vowel,,,,high,back,,"""
    )
)


def get_spectral_features():
    from transformers import AutoFeatureExtractor
    from whisperx import load_audio

    SAMPLING_RATE = 16000.0

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

    audiopath = "mats/black_audio.wav"
    audio = load_audio(audiopath)  # exactly 800 s
    n_chunks = np.ceil(audio.size / (30 * SAMPLING_RATE))  # 30 s each
    chunks = np.array_split(audio, n_chunks)
    features = feature_extractor(chunks, sampling_rate=SAMPLING_RATE)
    features = np.hstack(features["input_features"])

    chunks = np.array_split(features, TRS, axis=1)
    features = np.hstack([c.mean(axis=1, keepdims=True) for c in chunks])
    features = features.T

    return features


def get_transcript_features():
    df = pd.read_csv("mats/black_transcript.csv")
    df["TR"] = df.onset.divide(TR).apply(np.floor).apply(int)

    word_onsets = np.zeros(TRS, dtype=np.float32)
    word_rates = np.zeros(TRS, dtype=np.float32)
    for tr in range(TRS):
        subdf = df[df.TR == tr]
        if len(subdf):
            word_onsets[tr] = 1
            word_rates[tr] = len(subdf)

    return word_onsets, word_rates


def get_phoneme_features():
    from nltk.corpus import cmudict

    arpabet = cmudict.dict()
    # phone_set = ARPABET_PHONES
    # phonedict = {ph: i for i, ph in enumerate(phone_set)}

    # Build articulatory features of phonemes
    features = dfp.fillna("NA").iloc[:, 2:].values.flatten()
    feature_set = {f: i for i, f in enumerate(np.unique(features[features != "NA"]))}
    phone_embs = {}
    for i, row in dfp.iterrows():
        emb = np.zeros(len(feature_set))
        for feature in row.iloc[2:].dropna():
            emb[feature_set[feature]] = 1
        phone_embs[row.phoneme] = emb

    # def get_word_phone_emb(word):
    #     emb = np.zeros(len(phone_set))
    #     if phones := arpabet.get(word.lower()):
    #         for phone in phones[0]:
    #             emb[phonedict[phone.strip("012")]] = 1
    #     return emb

    def get_word_phone_features(word):
        emb = np.zeros(len(feature_set))
        if phones := arpabet.get(word.lower()):
            for phone in phones[0]:
                emb += phone_embs[phone.strip("012")]
        return emb

    df = pd.read_csv("mats/black_transcript.csv")
    df["TR"] = df.onset.divide(TR).apply(np.floor).apply(int)

    phone_emb = df.word.str.strip(PUNCTUATION).apply(get_word_phone_features)
    embeddings = np.vstack(phone_emb.values)
    # embeddings = embeddings[:, embeddings.sum(0) > 0]
    df["embeddings"] = embeddings.tolist()

    phone_emb = np.zeros((TRS, embeddings.shape[1]), dtype=np.float32)
    phone_rates = np.zeros(TRS, dtype=np.float32)
    for tr in range(TRS):
        subdf = df[df.TR == tr]
        if len(subdf):
            phone_emb[tr] = np.vstack(subdf.embeddings.values).mean(0)
            phone_rates[tr] = len(subdf)

    return phone_rates, phone_emb


def get_lexical_embs():
    df = pd.read_pickle("mats/black_model-gpt2-xl_layer-36.pkl")
    df["TR"] = df.onset.divide(TR).apply(np.floor).apply(int)

    n_features = df.iloc[0].embedding.size

    # Loop through TRs
    embeddings = np.zeros((TRS, n_features), dtype=np.float32)
    for tr in range(TRS):
        subdf = df[df.TR == tr]
        if len(subdf):
            embeddings[tr] = subdf.embedding.mean(0)

    return embeddings


def get_llm_embs(modelname: str, layer=0):
    import h5py
    from transformers import AutoTokenizer

    hfmodelname = HFMODELS[modelname]

    df = pd.read_csv("mats/black_transcript.csv")
    df["TR"] = df.onset.divide(TR).apply(np.floor).apply(int)

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname)
    df.insert(0, "word_idx", df.index.values)
    df["hftoken"] = df.word.apply(tokenizer.tokenize)
    df = df.explode("hftoken", ignore_index=True)
    df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)

    with h5py.File(f"features/black/{modelname}/states.hdf5", "r") as f:
        states = f[f"layer{layer}"][1:, :]  # (seq_len, dim) - skip first embedding (n)

    n_features = states.shape[1]
    embeddings = np.zeros((TRS, n_features), dtype=np.float32)

    # Loop through TRs
    for tr in range(TRS):
        mask = (df.TR == tr).values.astype(bool)
        if mask.any():
            embeddings[tr] = states[mask].mean(0)

    return embeddings


def extract_llm_embs(modelname: str, device: str = "cpu"):
    """
    to download the models:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        token = 'hf_<TOKEN>'  # for llama only
        modelname = "meta-llama/Llama-2-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(modelname, token=token)
        model = AutoModelForCausalLM.from_pretrained(modelname, token=token)

    to extract:
        salloc --time=00:10:00 --gres=gpu:1 --mem=32G
        python code/black_encoding.py -m mistral-7b
    """
    import h5py
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hfmodelname = HFMODELS[modelname]

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    df = pd.read_csv("mats/black_transcript.csv")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname)
    df.insert(0, "word_idx", df.index.values)
    df["hftoken"] = df.word.apply(tokenizer.tokenize)
    df = df.explode("hftoken", ignore_index=True)
    df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)

    tokenids = [tokenizer.bos_token_id] + df.token_id.tolist()
    batch = torch.tensor([tokenids], dtype=torch.long, device=device)

    model = AutoModelForCausalLM.from_pretrained(hfmodelname)
    model = model.eval()
    model = model.to(device)

    with torch.no_grad():
        output = model(batch, labels=batch, output_hidden_states=True)
        states = output.hidden_states

        loss = output.loss
        logits = output.logits[0]

        logits_order = logits.argsort(descending=True, dim=-1)
        ranks = torch.eq(logits_order[:-1], batch[:, 1:].T).nonzero()[:, 1]

        probs = logits[:-1, :].softmax(-1)
        true_probs = probs[0, batch[0, 1:]]

        entropy = torch.distributions.Categorical(probs=probs).entropy()

    df["rank"] = ranks.numpy(force=True)
    df["true_prob"] = true_probs.numpy(force=True)
    df["entropy"] = entropy.numpy(force=True)

    metrics = dict(top1_acc=(df["rank"] == 0).mean(), perplexity=loss.exp().item())
    print(metrics)

    outdir = f"features/black/{modelname}"
    makedirs(outdir, exist_ok=True)

    with h5py.File(f"{outdir}/states.hdf5", "w") as f:
        for layer in range(len(states)):
            layer_states = states[layer].squeeze().numpy(force=True)
            f.create_dataset(name=f"layer{layer}", data=layer_states)

    with open(f"{outdir}/performance.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df.to_csv(f"{outdir}/transcript.csv")


def get_bold(sub: int) -> np.ndarray:
    boldpath = Path(
        root="data/derivatives/fmriprep/",
        datatype="func",
        sub=f"{sub:03d}",
        ses=1,
        task="Black",
        run=1,
        space="fsaverage6",
        hemi="L",
        suffix="bold",
        ext=".func.gii",
    )
    boldpath.update(sub=f"{sub:03d}")
    paths = [boldpath, boldpath.copy().update(hemi="R")]

    confpath = boldpath.copy()
    del confpath["hemi"]
    del confpath["space"]
    confpath.update(desc="confounds", suffix="timeseries", ext=".tsv")

    confounds = MOTION_CONFOUNDS + CONFOUND_REGRESSORS
    confdata = pd.read_csv(confpath, sep="\t", usecols=confounds)
    confdata.bfill(inplace=True)

    masker = subject.GiftiMasker(
        t_r=TR,
        ensure_finite=True,
        standardize="zscore_sample",
        standardize_confounds=True,
    )
    Y_bold = masker.fit_transform(paths, confounds=confdata.to_numpy())
    Y_bold = Y_bold[8:-8]

    return Y_bold


def build_regressors(modelname=None, layer=0):
    # lexical_embs = get_lexical_embs()
    lexical_embs = get_llm_embs(modelname=modelname, layer=layer)
    phone_rates, phone_embs = get_phoneme_features()
    word_onsets, word_rates = get_transcript_features()
    spectral_features = get_spectral_features()

    X = np.hstack(
        (
            word_onsets.reshape(-1, 1),
            word_rates.reshape(-1, 1),
            phone_rates.reshape(-1, 1),
            spectral_features,
            phone_embs,
            lexical_embs,
        )
    )

    slices = {
        "task": slice(0, 3),
        "spectral": slice(3, 3 + 80),
        "phonetic": slice(3 + 80, 3 + 80 + 22),
        "lexical": slice(3 + 80 + 22, X.shape[1]),
    }

    return X, slices


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    verbose: int,
    n_jobs: int,
    delayer_class=Delayer,
):
    """Build the pipeline"""

    # Set up modeling pipeline
    delayer_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        delayer_class(delays=[2, 3, 4, 5]),
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
        mkr_model,
    )

    return pipeline


def main(args):
    layer = args.layer

    X, features = build_regressors(modelname=args.model, layer=layer)
    feature_names = list(features.keys())
    slices = list(features.values())

    pipeline = build_model(feature_names, slices, args.alphas, args.verbose, args.jobs)

    # not all subjects have this..
    subs = list(SUBS_STRANGERS)
    subs.remove(11)
    subs.remove(111)
    subs.remove(12)

    cache_path = Path(
        root="data/derivatives/cleaned",
        datatype="func",
        sub="000",
        task="Black",
        space="fsaverage6",
        suffix="bold",
        ext="hdf5",
    )

    for sub in tqdm(subs):
        # get BOLD data
        cache_path.update(sub=f"{sub:03d}")
        if cache_path.isfile():
            with h5py.File(cache_path, "r") as f:
                Y_bold = f["bold"][...]
        else:
            Y_bold = get_bold(sub)
            with h5py.File(cache_path, "w") as f:
                f.create_dataset(name="bold", data=Y_bold)

        K = 2
        results = defaultdict(list)
        kfold = KFold(n_splits=K)
        for train_index, test_index in tqdm(kfold.split(X), leave=False, total=K):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]

            pipeline.fit(X_train, Y_train)

            Y_preds = pipeline.predict(X_test, split=True)
            scores_split = correlation_score_split(Y_test, Y_preds)

            results["cv_scores"].append(scores_split.numpy(force=True))
            results["cv_preds"].append(Y_preds.numpy(force=True))

        # stack across folds
        result = {k: np.stack(v) for k, v in results.items()}

        # save
        pklpath = Path(
            root="encoding/black",
            sub=f"{sub:03d}",
            datatype=f"model-{args.model}_layer-{layer}",
            ext=".hdf5",
        )
        pklpath.mkdirs()
        with h5py.File(pklpath, "w") as f:
            for key, value in result.items():
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

    print(args)
    main(args)
    # extract_llm_embs(modelname=args.model)
