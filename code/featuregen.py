""" """

from glob import glob

import numpy as np
import pandas as pd
from constants import (
    ARPABET_PHONES,
    CONVS_FRIENDS,
    CONVS_STRANGERS,
    FNKEYS,
    PUNCTUATION,
    RUN_TRIAL_SLICE,
    RUNS,
    SUBS_FRIENDS,
    SUBS_STRANGERS,
    TRIALS,
)
from tqdm import tqdm
from util.path import Path
from util.subject import get_trials


def confounds(args):
    confpath = Path(
        root="data/derivatives/fmriprep",
        sub="000",
        ses="1",
        datatype="func",
        task="Conv",
        run=0,
        desc="confounds",
        suffix="timeseries",
        ext=".tsv",
    )

    confounds = []  # NOTE removed this constant
    print(len(confounds))

    outpath = Path(
        root="features", sub="000", datatype="motion", run=0, trial=0, ext="npy"
    )

    for sub in tqdm(args.subs):
        substr = f"{sub:03d}"
        confpath.update(sub=substr)
        rt_dict = get_trials(sub, condition="G")
        for run in args.runs:
            confpath.update(run=run)
            trials = rt_dict[run]
            for trial in trials:
                df = pd.read_csv(confpath, sep="\t", usecols=confounds)
                conv_slice = RUN_TRIAL_SLICE[trial]
                conf_data = df.iloc[conv_slice].to_numpy()

                outpath.update(sub=substr, run=run, trial=trial)
                outpath.mkdirs()
                np.save(outpath, conf_data)


def phonemes(args, mode="articulatory"):
    from io import StringIO

    from nltk.corpus import cmudict

    arpabet = cmudict.dict()
    phone_set = ARPABET_PHONES
    phonedict = {ph: i for i, ph in enumerate(phone_set)}

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

    # Build articulatory features of phonemes
    features = dfp.fillna("NA").iloc[:, 2:].values.flatten()
    feature_set = {f: i for i, f in enumerate(np.unique(features[features != "NA"]))}
    phone_embs = {}
    for i, row in dfp.iterrows():
        emb = np.zeros(len(feature_set))
        for feature in row.iloc[2:].dropna():
            emb[feature_set[feature]] = 1
        phone_embs[row.phoneme] = emb

    def get_word_phone_features(word):
        emb = np.zeros(len(feature_set))
        if phones := arpabet.get(word.lower()):
            for phone in phones[0]:
                emb += phone_embs[phone.strip("012")]
        return emb

    def get_word_phone_emb(word):
        emb = np.zeros(len(phone_set))
        if phones := arpabet.get(word.lower()):
            for phone in phones[0]:
                emb[phonedict[phone.strip("012")]] = 1
        return emb

    func = get_word_phone_emb
    if mode == "articulatory":
        func = get_word_phone_features

    # Look for transcripts
    transpath = Path(root="data/stimuli", datatype="whisperx", ext=".csv")
    transpath.update(**{str(k): v for k, v in vars(args).items() if k in FNKEYS})
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    for filename in tqdm(files):
        transpath = Path.frompath(filename)
        transpath.update(root="data/stimuli", datatype="whisperx")

        df = pd.read_csv(transpath)
        phone_emb = df.word.astype(str).str.strip(PUNCTUATION).apply(func)
        embeddings = np.vstack(phone_emb.values)

        # remove any uninformative dimensions or not
        # embeddings = embeddings[:, embeddings.sum(0) > 0]
        df["embedding"] = embeddings.tolist()

        transpath.update(root="data/stimuli", datatype=mode, ext=".pkl")
        transpath.mkdirs()
        df.to_pickle(transpath)


def syntactic(args):
    import spacy
    from sklearn.preprocessing import LabelBinarizer

    nlp = spacy.load("en_core_web_lg")

    taggerEncoder = LabelBinarizer().fit(nlp.get_pipe("tagger").labels)
    dependencyEncoder = LabelBinarizer().fit(nlp.get_pipe("parser").labels)

    # Look for transcripts
    transpath = Path(root="data/stimuli", datatype="whisperx", ext=".csv")
    transpath.update(**{str(k): v for k, v in vars(args).items() if k in FNKEYS})
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    # Process transcripts
    for filename in tqdm(files):
        transpath = Path.frompath(filename)
        transpath.update(root="data/stimuli", datatype="whisperx")

        df = pd.read_csv(transpath)
        df["turn"] = (df.speaker.diff() != 0).cumsum()

        df.insert(0, "word_idx", df.index.values)
        df["word_with_ws"] = df.word.astype(str) + " "
        try:
            df["hftoken"] = df.word_with_ws.apply(nlp.tokenizer)
        except TypeError:
            print("typeerror!")
            breakpoint()
        df = df.explode("hftoken", ignore_index=True)

        features = []
        for _, sentence in df.groupby(["speaker", "turn"]):
            # create a doc from the pre-tokenized text then parse it for features
            words = [token.text for token in sentence.hftoken.tolist()]
            spaces = [token.whitespace_ == " " for token in sentence.hftoken.tolist()]
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
        embeddings = np.hstack((a, b, c))
        df["embedding"] = embeddings.tolist()

        # not serializable
        df.drop(["hftoken", "word_with_ws"], axis=1, inplace=True)

        transpath.update(root="data/stimuli", datatype="syntactic", ext=".pkl")
        transpath.mkdirs()
        df.to_pickle(transpath)


def spacy_vectors(args):
    import spacy

    nlp = spacy.load("en_core_web_lg")

    # Look for transcripts
    transpath = Path(
        root="stimuli", datatype="transcript", suffix="aligned", ext=".csv"
    )
    transpath.update(**{str(k): v for k, v in vars(args).items() if k in FNKEYS})
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    # Process transcripts
    for filename in tqdm(files):
        transpath = Path.frompath(filename)
        transpath.update(root="stimuli", datatype="transcript")

        df = pd.read_csv(transpath)

        df.insert(0, "word_idx", df.index.values)
        df["word_with_ws"] = df.word.astype(str) + " "
        try:
            df["hftoken"] = df.word_with_ws.apply(nlp.tokenizer)
        except TypeError:
            print("typeerror!")
            breakpoint()
        df = df.explode("hftoken", ignore_index=True)

        features = []

        for _, sentence in df.groupby(["speaker", "sentence"]):
            # create a doc from the pre-tokenized text then parse it for features
            words = [token.text for token in sentence.hftoken.tolist()]
            spaces = [token.whitespace_ == " " for token in sentence.hftoken.tolist()]
            doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
            doc = nlp(doc)
            for token in doc:
                features.append(token.vector)

        df.drop(["hftoken", "word_with_ws"], axis=1, inplace=True)  # not serializable
        df["embedding"] = features

        transpath.update(root="features", datatype="en_core_web_lg", ext=".pkl")
        transpath.mkdirs()
        df.to_pickle(transpath)


def bag_of_words(args):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Look for transcripts

    for conv in tqdm(CONVS_STRANGERS):
        transcripts = f"data/stimuli/conv-{conv}/whisperx/*.csv"
        files = glob(transcripts)
        assert len(files), "No files found for: " + transcripts

        dfs = [pd.read_csv(filename) for filename in files]
        all_words = sum([df["word"].to_list() for df in dfs], [])
        vectorizer = TfidfVectorizer().fit(all_words)

        for filename in files:
            transcript_df = pd.read_csv(filename)
            vectors = vectorizer.transform(transcript_df["word"].to_list())
            vectors = vectors.toarray()
            transcript_df["embedding"] = vectors.tolist()

            transpath = Path.frompath(filename)
            transpath.update(root="data/stimuli", datatype="bow", ext=".pkl")
            transpath.mkdirs()
            transcript_df.to_pickle(transpath)


def wordnet(args):
    """
    https://github.com/nlx-group/WordNetEmbeddings?tab=readme-ov-file

    Saedi, Chakaveh, António Branco, João António Rodrigues and João Ricardo
    Silva, 2018, "WordNet Embeddings", In Proceedings, 3rd Workshop on
    Representation Learning for Natural Language Processing (RepL4NLP), 56th
    Annual Meeting of the Association for Computational Linguistics, 15-20 July
    2018, Melbourne, Australia.
    """

    import string

    from nltk.stem import WordNetLemmatizer
    from nltk.stem.porter import PorterStemmer

    wnl = WordNetLemmatizer()
    port = PorterStemmer()

    # Load wordnet embeddings
    emb_dict = {}
    with open("mats/wn2vec.txt", "r") as f:
        first_line = f.readline()
        total, _ = first_line.split()

        for line in tqdm(f, total=int(total)):
            word, vecstr = line.strip().split(maxsplit=1)
            vector = np.fromstring(vecstr, dtype=float, sep=" ")
            emb_dict[word] = vector

    # Look for transcripts
    transpath = Path(
        root="stimuli", datatype="transcript", suffix="aligned", ext=".csv"
    )
    transpath.update(**{str(k): v for k, v in vars(args).items() if k in FNKEYS})
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    def find_wordnet_vec(word: str):
        word = word.strip(string.punctuation).lower()

        lemma = wnl.lemmatize(word)
        vector = emb_dict.get(lemma)
        if vector is None:
            stem = port.stem(word)
            vector = emb_dict.get(stem)

        return vector

    # Process transcripts
    for filename in tqdm(files):
        transpath = Path.frompath(filename)
        transpath.update(root="stimuli", datatype="transcript")

        df = pd.read_csv(transpath)
        df.insert(0, "word_idx", df.index.values)
        embeddings = df.word.apply(find_wordnet_vec)
        df["embedding"] = embeddings.tolist()

        transpath.update(root="features", datatype="wordnet", ext=".pkl")
        transpath.mkdirs()
        df.to_pickle(transpath)


def spectral(args):
    """Move and process transcripts."""
    from transformers import AutoFeatureExtractor
    from whisperx import load_audio

    SAMPLING_RATE = 16000.0

    # Look for audio files
    audiopath = Path(root="data/stimuli", datatype="audio", ext=".wav")
    audiopath.update(**{str(k): v for k, v in vars(args).items() if k in FNKEYS})
    search_str = audiopath.starstr(["conv", "datatype"])
    search_str = search_str.replace(".wav", "condition-G*.wav")
    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

    for filename in tqdm(files):
        audiopath = Path.frompath(filename)
        audiopath.update(root="data/stimuli", datatype="audio")

        audio = load_audio(audiopath)
        chunks = np.array_split(audio, 6)  # six 30-second chunks
        features = feature_extractor(chunks, sampling_rate=SAMPLING_RATE)
        features = np.hstack(features["input_features"])

        chunks = np.split(features, 120, axis=1)  # 150 10-ms chunks (1.5 s)
        features = np.hstack([c.mean(axis=1, keepdims=True) for c in chunks])

        audiopath.update(datatype="spectrogram", ext="npy")
        audiopath.mkdirs()
        np.save(audiopath, features.T)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("feature", type=str)
    parser.add_argument("-s", "--subs", type=str, default=SUBS_STRANGERS)
    parser.add_argument("-c", "--convs", type=str, default=CONVS_STRANGERS)
    parser.add_argument("-r", "--runs", nargs="*", type=int, default=RUNS)
    parser.add_argument("-t", "--trials", nargs="*", type=int, default=TRIALS)
    args = parser.parse_args()

    if args.convs == "strangers":
        args.convs = CONVS_STRANGERS
    elif args.convs == "friends":
        args.convs = CONVS_FRIENDS

    if args.subs == "strangers":
        args.subs = SUBS_STRANGERS
    elif args.subs == "friends":
        args.subs = SUBS_FRIENDS

    if args.feature == "spectral":
        spectral(args)
    elif args.feature == "articulatory":
        phonemes(args, mode="articulatory")
    elif args.feature == "phonemic":
        phonemes(args, mode="phonemic")
    elif args.feature == "confounds":
        confounds(args)
    elif args.feature == "wordnet":
        wordnet(args)
    elif args.feature == "syntactic":
        syntactic(args)
    elif args.feature == "spacy":
        spacy_vectors(args)
    elif args.feature == "bow":
        bag_of_words(args)
    else:
        raise ValueError(f"Unknown feature set: {args.feature}")
