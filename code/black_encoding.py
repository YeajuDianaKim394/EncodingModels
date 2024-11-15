"""Story encoding"""

import json
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from io import StringIO

import encoding
import ffmpeg
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from clean import DEFAULT_CONFOUND_MODEL
from constants import PUNCTUATION, SUBS_STRANGERS, TR
from embeddings import HFMODELS
from himalaya.backend import set_backend
from himalaya.kernel_ridge import (ColumnKernelizer, Kernelizer,
                                   MultipleKernelRidgeCV)
from himalaya.scoring import correlation_score_split
from nilearn import signal
from nltk.corpus import cmudict
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from util import subject
from util.extract_confounds import extract_confounds, load_confounds
from util.path import Path
from voxelwise_tutorials.delayer import Delayer

TRS = 534
# the first 12 seconds is the ready screen (8 TRs), 13:20 (800 s) the actual
# story (534 TRs), and last 12 seconds (8 TRs)  done screen.

RESERVED_NAMES = ["contextual", "syntactic", "articulatory", "spectral", "static"]

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

    SAMPLING_RATE = 16000.0

    # from whisperx load_audio function.
    def load_audio(file: str, sr: int = SAMPLING_RATE):
        """
        Open an audio file and read as mono waveform, resampling as necessary

        Parameters
        ----------
        file: str
            The audio file to open

        sr: int
            The sample rate to resample the audio if necessary

        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

    audiopath = "data/stimuli/black/black_audio.wav"
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
    df = pd.read_csv("data/stimuli/black/black_transcript.csv")
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

    df = pd.read_csv("data/stimuli/black/black_transcript.csv")
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


def get_syntactic_features():
    import spacy
    from sklearn.preprocessing import LabelBinarizer

    nlp = spacy.load(
        "en_core_web_lg", exclude=["toke2vec", "attribute_ruler", "lemmatizer", "ner"]
    )

    taggerEncoder = LabelBinarizer().fit(nlp.get_pipe("tagger").labels)
    dependencyEncoder = LabelBinarizer().fit(nlp.get_pipe("parser").labels)

    with open("data/stimuli/black/black_audio.json", "r") as f:
        d = json.load(f)
        df = pd.DataFrame(d["segments"])
        df.insert(0, "segment_id", df.index.values)
        df = df.explode("words")
        df["word"] = df.words.apply(lambda x: x["word"])
        df["score"] = df.words.apply(lambda x: x.get("score", np.nan))
        df["onset"] = df.words.apply(lambda x: x.get("start", np.nan))
        df["offset"] = df.words.apply(lambda x: x.get("end", np.nan))
        df.drop("words", axis=1, inplace=True)
        df.drop("text", axis=1, inplace=True)
        df.ffill(inplace=True)

    df["TR"] = df.onset.divide(TR).apply(np.floor).apply(int)

    df.insert(0, "word_idx", df.index.values)
    df["word_with_ws"] = df.word.astype(str) + " "
    try:
        df["hftoken"] = df.word_with_ws.apply(nlp.tokenizer)
    except TypeError:
        print("typeerror!")
        breakpoint()
    df = df.explode("hftoken", ignore_index=True)

    features = []

    for _, sentence in df.groupby(["segment_id"]):
        # create a doc from the pre-tokenized text then parse it for features
        words = [token.text for token in sentence.hftoken.tolist()]
        spaces = [token.whitespace_ for token in sentence.hftoken.tolist()]
        doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
        doc = nlp(doc)
        for token in doc:
            features.append([token.text, token.tag_, token.dep_, token.is_stop])

    df2 = pd.DataFrame(
        features, columns=["token", "pos", "dep", "stop"], index=df.index
    )
    df = pd.concat([df, df2], axis=1)

    # generate embeddings
    a = taggerEncoder.transform(df.pos.tolist())
    b = dependencyEncoder.transform(df.dep.tolist())
    c = LabelBinarizer().fit_transform(df.stop.tolist())
    features = np.hstack((a, b, c))
    # remove any uninformative dimensions or not
    if np.any(features.sum(0) == 0):
        features = features[:, features.sum(0) > 0]
    # df["embedding"] = embeddings.tolist()

    # not serializable
    df.drop(["hftoken", "word_with_ws"], axis=1, inplace=True)

    n_features = features.shape[1]
    embeddings = np.zeros((TRS, n_features), dtype=np.float32)

    # Loop through TRs
    for tr in range(TRS):
        mask = (df.TR == tr).values.astype(bool)
        if mask.any():
            embeddings[tr] = features[mask].mean(0)

    return embeddings


def get_llm_embs(modelname: str, layer=0):

    df = pd.read_csv(f"data/stimuli/black/{modelname}_transcript.csv")
    df["onset"] = df["start"]
    df.ffill(inplace=True)
    df["TR"] = df["onset"].divide(TR).apply(np.floor).apply(int)

    with h5py.File(f"data/stimuli/black/{modelname}_states.hdf5", "r") as f:
        states = f[f"layer-{layer}"][...]  # (seq_len, dim) - skip first embedding (n)

    n_features = states.shape[1]
    embeddings = np.zeros((TRS, n_features), dtype=np.float32)

    # Loop through TRs
    for tr in range(TRS):
        mask = (df.TR == tr).values.astype(bool)
        if mask.any():
            embeddings[tr] = states[mask].mean(0)

    return embeddings


def extract_llm_embs(model: str, device: str = "cpu", context_len: int = 128, **kwargs):
    from accelerate import Accelerator, find_executable_batch_size
    from transformers import AutoModelForCausalLM, AutoTokenizer

    modelname = model
    hfmodelname = HFMODELS.get(model, model)
    output_dir = "data/stimuli/black"

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    with open("data/stimuli/black/black_audio.json", "r") as f:
        d = json.load(f)
        df = pd.DataFrame(d["word_segments"])

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname)
    df.insert(0, "word_idx", df.index.values)
    df["hftoken"] = df.word.apply(lambda x: tokenizer.tokenize(" " + x))
    df = df.explode("hftoken", ignore_index=True)
    df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)

    model = AutoModelForCausalLM.from_pretrained(hfmodelname)
    model = model.eval()
    model = model.to(device)

    fill_value = 0
    if tokenizer.pad_token_id is not None:
        fill_value = tokenizer.pad_token_id

    token_ids = df.token_id.tolist()
    data = torch.full((len(token_ids), context_len + 1), fill_value, dtype=torch.long)
    for i in range(len(token_ids)):
        example_tokens = token_ids[max(0, i - context_len) : i + 1]
        data[i, -len(example_tokens) :] = torch.tensor(example_tokens)

    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=32)
    def inference_loop(batch_size=32):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        # accelerator.print(f"Trying batch size: {batch_size}")

        data_dl = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False
        )

        top_guesses = []
        ranks = []
        true_probs = []
        entropies = []
        embeddings = []

        with torch.no_grad():
            for batch in data_dl:
                output = model(batch.to(device), output_hidden_states=True)
                logits = output.logits
                states = output.hidden_states

                true_ids = batch[:, -1]
                brange = list(range(len(true_ids)))
                logits_order = logits[:, -2, :].argsort(descending=True)
                batch_top_guesses = logits_order[:, 0]
                batch_ranks = torch.eq(
                    logits_order, true_ids.reshape(-1, 1).to(device)
                ).nonzero()[:, 1]
                batch_probs = torch.softmax(logits[:, -2, :], dim=-1)
                batch_true_probs = batch_probs[brange, true_ids]
                batch_entropy = torch.distributions.Categorical(
                    probs=batch_probs
                ).entropy()
                batch_embeddings = [
                    state[:, -1, :].numpy(force=True) for state in states
                ]

                top_guesses.append(batch_top_guesses.numpy(force=True))
                ranks.append(batch_ranks.numpy(force=True))
                true_probs.append(batch_true_probs.numpy(force=True))
                entropies.append(batch_entropy.numpy(force=True))
                embeddings.append(batch_embeddings)

            return top_guesses, ranks, true_probs, entropies, embeddings

    top_guesses, ranks, true_probs, entropies, embeddings = inference_loop()

    df["rank"] = np.concatenate(ranks)
    df["true_prob"] = np.concatenate(true_probs)
    df["top_pred"] = np.concatenate(top_guesses)
    df["entropy"] = np.concatenate(entropies)

    df.to_csv(f"{output_dir}/{modelname}_transcript.csv")

    with h5py.File(f"{output_dir}/{modelname}_states.hdf5", "w") as f:
        for i in range(len(embeddings[0])):
            layer_embeddings = np.vstack([e[i] for e in embeddings])
            f.create_dataset(name=f"layer-{i}", data=layer_embeddings)

    with open(f"{output_dir}/{modelname}_performance.json", "w") as f:
        json.dump(dict(top1_accuracy=(df["rank"] == 0).mean()), f)

    # with h5py.File(f"{output_dir}/{modelname}_states.hdf5", "w") as f:
    #     for layer in range(len(states)):
    #         layer_states = states[layer].squeeze().numpy(force=True)
    #         f.create_dataset(name=f"layer{layer}", data=layer_states)


def get_bold(sub: int) -> np.ndarray:

    bold_path = Path(
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
    bold_path.update(sub=f"{sub:03d}")

    confpath = bold_path.copy()
    del confpath["hemi"]
    del confpath["space"]
    confpath.update(desc="confounds", suffix="timeseries", ext=".tsv")

    confounds_df, confounds_meta = load_confounds(confpath)
    confounds_df.bfill(inplace=True)  # fill in nans when using derivatives
    confounds = extract_confounds(confounds_df, confounds_meta, DEFAULT_CONFOUND_MODEL)

    hemi_bold = []
    for hemi in ["L", "R"]:
        bold_path.update(hemi=hemi)
        img = nib.load(bold_path)
        run_bold = img.agg_data()
        hemi_bold.append(run_bold)
    bold = np.vstack(hemi_bold)

    cleaned_bold = signal.clean(
        bold.T,
        confounds=confounds,
        detrend=True,
        t_r=TR,
        ensure_finite=True,
        standardize="zscore_sample",
        standardize_confounds=True,
    )

    cleaned_bold = cleaned_bold[8:-8]

    return cleaned_bold


def build_regressors(modelname: str, layer: int):

    hf_name = "gpt2-2b" if modelname in RESERVED_NAMES else modelname

    lexical_embs = get_llm_embs(modelname=hf_name, layer=layer)
    phone_rates, phone_embs = get_phoneme_features()
    word_onsets, word_rates = get_transcript_features()
    spectral_features = get_spectral_features()

    features = [
        word_onsets.reshape(-1, 1),
        word_rates.reshape(-1, 1),
        phone_rates.reshape(-1, 1),
    ]

    dim_nuisance = 3
    dim_lexical = lexical_embs.shape[1]
    dim_phonemic = phone_embs.shape[1]
    dim_spectral = spectral_features.shape[1]

    slices = {
        "task": slice(0, dim_nuisance),
    }

    if modelname == "contextual":
        features.append(lexical_embs)
        slices["lexical"] = slice(dim_nuisance, dim_lexical)
    elif modelname == "syntactic":
        syntactic_embs = get_syntactic_features()
        dim_syntactic = syntactic_embs.shape[1]
        features.append(syntactic_embs)
        slices["syntactic"] = slice(dim_nuisance, dim_syntactic)
    elif modelname == "articulatory":
        features.append(phone_embs)
        slices["articulatory"] = slice(dim_nuisance, dim_phonemic)
    elif modelname == "spectral":
        features.append(spectral_features)
        slices["spectral"] = slice(dim_nuisance, dim_spectral)
    else:
        features.append(spectral_features)
        features.append(phone_embs)
        features.append(lexical_embs)

        slices["spectral"] = slice(dim_nuisance, dim_nuisance + dim_spectral)
        slices["phonetic"] = slice(
            slices["spectral"].stop, slices["spectral"].stop + dim_phonemic
        )
        slices["lexical"] = slice(
            slices["phonetic"].stop, slices["phonetic"].stop + dim_lexical
        )

    X = np.hstack(features)
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
        StandardScaler(with_mean=True, with_std=True),
        delayer_class(delays=[0, 1, 2, 3, 4, 5]),
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


def encoding_story_to_conv(
    model: str,
    layer: int,
    verbose: bool,
    n_jobs: int,
    alphas: np.ndarray,
):

    X, features = build_regressors(modelname=model, layer=layer)
    feature_names = list(features.keys())
    slices = list(features.values())
    print(feature_names)
    print(slices)

    pipeline = build_model(feature_names, slices, alphas, verbose, n_jobs)

    delayer = encoding.SplitDelayer(delays=[0, 1, 2, 3, 4, 5])

    # not all subjects have this..
    subs = list(SUBS_STRANGERS)
    subs.remove(11)
    subs.remove(111)
    subs.remove(12)

    # cache_path = Path(
    #     root="data/derivatives/clean",
    #     datatype="func",
    #     sub="000",
    #     task="black",
    #     space="fsaverage6",
    #     suffix="bold",
    #     ext="hdf5",
    # )
    # cache_path.update(sub=f"{sub:03d}")
    # if cache_path.isfile():
    #     with h5py.File(cache_path, "r") as f:
    #         Y_bold = f["bold"][...]
    # else:
    #     Y_bold = get_bold(sub)
    #     with h5py.File(cache_path, "w") as f:
    #         f.create_dataset(name="bold", data=Y_bold)

    hf_name = "gpt2-2b" if model in RESERVED_NAMES else model
    datatype = f"model-{hf_name}_layer-{layer}"

    for sub in tqdm(subs):

        Y_bold = get_bold(sub)
        # NOTE
        spaces = encoding.SPACES["joint_nosplit"]
        # spaces = encoding.SPACES["llm_nosplit"]
        conv_X, _ = encoding.build_regressors(sub, datatype, spaces=spaces, split=False)
        conv_bold = subject.get_bold(sub, cache="default_task")

        print(
            datetime.now(),
            sub,
            X.shape,
            Y_bold.shape,
            conv_X.shape,
            conv_bold.shape,
        )

        # fit on story
        pipeline.fit(X, Y_bold)
        # test on conversation
        Y_preds = pipeline.predict(conv_X[:, 2:], split=True)

        prod_mask = conv_X[:, 0:1]
        comp_mask = conv_X[:, 1:2]

        # inclusive scoring
        prod_mask = delayer.fit_transform(prod_mask).any(-1)
        comp_mask = delayer.fit_transform(comp_mask).any(-1)
        scores_prod = correlation_score_split(
            conv_bold[prod_mask], Y_preds[:, prod_mask, :]
        )
        scores_comp = correlation_score_split(
            conv_bold[comp_mask], Y_preds[:, comp_mask, :]
        )

        results = dict()
        results["cv_scores_prod"] = scores_prod.numpy(force=True)
        results["cv_scores_comp"] = scores_comp.numpy(force=True)
        results["cv_preds"] = Y_preds.numpy(force=True)

        # exclusive
        prod_only_mask = prod_mask & ~comp_mask
        comp_only_mask = comp_mask & ~prod_mask
        scores_prod = correlation_score_split(
            conv_bold[prod_only_mask], Y_preds[:, prod_only_mask, :]
        )
        scores_comp = correlation_score_split(
            conv_bold[comp_only_mask], Y_preds[:, comp_only_mask, :]
        )
        results["cv_scores_prod_exclusive"] = scores_prod.numpy(force=True)
        results["cv_scores_comp_exclusive"] = scores_comp.numpy(force=True)

        # encoding within black story
        cv_results = defaultdict(list)
        kfold = KFold(n_splits=2)
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]
            pipeline.fit(X_train, Y_train)
            Y_preds = pipeline.predict(X_test, split=True)
            scores_split = correlation_score_split(Y_test, Y_preds)
            cv_results["black_cv_scores"].append(scores_split.numpy(force=True))
            cv_results["black_cv_preds"].append(Y_preds.numpy(force=True))
        for k, v in cv_results.items():
            results[k] = np.stack(v)

        # save
        pklpath = Path(
            root="results/encoding_black",
            sub=f"{sub:03d}",
            datatype=model,
            ext=".hdf5",
        )
        pklpath.mkdirs()
        with h5py.File(pklpath, "w") as f:
            for key, value in results.items():
                f.create_dataset(name=key, data=value)


def main(extract_only: bool, cuda: int, **kwargs):
    if cuda > 0:
        if torch.cuda.is_available():
            set_backend("torch_cuda")
        else:
            print("[WARN] cuda not available")

    if extract_only:
        extract_llm_embs(**kwargs)
    else:
        encoding_story_to_conv(**kwargs)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="gpt2-2b")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    _args = parser.parse_args()
    _args.alphas = np.logspace(0, 12, 13)

    main(**vars(_args))
