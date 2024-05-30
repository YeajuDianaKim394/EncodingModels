"""Prepare transcripts for forced alignment with audio.

Creates a normalized word-level .csv and utterance-level .TextGrid for each transcript.

Pipline
-------

1. Tokenize utterances and segment into sentences
    0. uses vocab from MFA dictionary
1. Normalize each token
    0. remove punctuation, white space, and lower case
1. Save CSV with the following columns:
    0. speaker, utt_id, utt_onset, utt_offset, sent_id, token_id, word, word_normalized
    0. ensures that joining all the words will return the original (cleaned) text
1. Create a TextGrid with an Tier per speaker using the joined word_normalized column per utt_id group

https://github.com/facebookresearch/fairseq/blob/main/examples/mms/data_prep/text_normalization.py
https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/montreal_forced_aligner/corpus/multiprocessing.py#L260
https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html#text-normalization-and-dictionary-lookup
"""

import re
from glob import glob

import pandas as pd
import spacy
from constants import CONVS_FRIENDS, CONVS_STRANGERS, FNKEYS, PUNCTUATION
from librosa import get_duration
from spacy.symbols import ORTH
from util.path import Path
from util.transcription import records2tg


def utterance2words(uttdf: pd.DataFrame, nlp: spacy.language.Language) -> pd.DataFrame:
    """Transform dataframe from utterance- to word-level.

    This will also infer sentence ids.
    """

    records = list(uttdf.to_dict(orient="index").values())

    # Get individual sentences and tokens
    for entry in records:
        text = entry["text"]
        doc = nlp(text)
        tokens = []
        sentences = []
        punctuations = []
        for i, sent in enumerate(doc.sents):
            for token in sent:
                sentences.append(i)
                tokens.append(token.text_with_ws)
                punctuations.append(token.is_punct)
        entry["sentence_id"] = sentences
        entry["is_punct"] = punctuations
        entry["token"] = tokens

    # Re-aggregate into DataFrame
    df = pd.DataFrame(records)
    df = df.explode(["sentence_id", "is_punct", "token"], ignore_index=True)
    df.drop("text", axis=1, inplace=True)
    df["is_punct"] = df.is_punct.astype(bool)
    df["token_norm"] = df.token.str.strip().str.lower().str.strip(PUNCTUATION)

    # for debugging and testing:
    # sents = df.groupby(['speaker', 'turn', 'sentence_id']).token.apply(''.join)

    return df


def get_spacy() -> spacy.language.Language:
    """Setup a spacy pipeline for tokenization and sentencization.

    https://spacy.io/usage/linguistic-features#tokenization
    https://spacy.io/usage/linguistic-features#native-tokenizer-additions
    """

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    # Don't tokenizer words we have in the acoustic dictionary
    df = pd.read_csv("english_mfa.dict", sep="\t", header=None, usecols=[0])
    vocab = set(df.values.flatten().tolist())
    punc_sub = re.compile(r"[’‘]")
    nlp.tokenizer.rules = {
        key: value
        for key, value in nlp.tokenizer.rules.items()
        if punc_sub.sub("'", key.lower()) not in vocab
        and "'" not in key
        and "’" not in key
        and "‘" not in key
    }

    # We want to preserve suffixes like 's because the acoustic model can handle it
    suffixes = list(nlp.Defaults.suffixes)  # type: ignore
    suffixes.remove("'s")
    suffix_re = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_re.search

    # non-whitespace separators
    infixes = nlp.Defaults.infixes + [r"([\[\]&:])"]  # type: ignore
    infix_re = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    # add special case tags
    nlp.tokenizer.add_special_case("[laughter]", [{ORTH: "[laughter]"}])
    nlp.tokenizer.add_special_case("[inaudible]", [{ORTH: "[inaudible]"}])

    # For debugging and testing
    # print([t.text for t in nlp('hello- world. 00:00; M&M; [laughter] and [inaudible]')])
    # breakpoint()

    return nlp


def convert_wdf_tg(df: pd.DataFrame):
    """Convert word-level dataframe to TextGrid."""
    turns = (
        df[~df.is_punct]
        .groupby("utterance")
        .agg(
            {
                "speaker": "first",
                "token_norm": " ".join,  # TODO - don't use a space(?)
                "onset": "first",
                "offset": "last",
            }
        )
    )
    utt_records = []
    for _, row in turns.iterrows():
        utt_records.append(
            {
                "speaker": row.speaker,
                "onset": row.onset,
                "offset": row.offset,
                "text": row.token_norm,
            }
        )
    return records2tg(utt_records)


def main(args):
    """Move and process transcripts."""

    # Look for transcripts
    transpath = Path(
        root="stimuli", datatype="transcript", suffix="utterance", ext=".csv"
    )
    transpath.update(**{str(k): v for k, v in vars(args).items() if k in FNKEYS})
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    nlp = get_spacy()
    for transfn in files:
        transpath = Path.frompath(transfn)
        transpath.update(root="stimuli", datatype="transcript")

        # conv_id = transpath["conv"]
        # if conv_id not in CONVS_FRIENDS:  # NOTE - temporary
        #     continue

        uttdf = pd.read_csv(transfn)

        # Add offset and turn information
        uttdf.insert(
            2, "offset", uttdf.onset.shift(-1, fill_value=180)
        )  # 180 s per trial
        uttdf.reset_index(names="utterance", inplace=True)
        turns = (uttdf.speaker.diff().abs().fillna(0).cumsum() / 100).astype(int)
        uttdf.insert(0, "turn", turns)

        # # Add actual offset from wav file (not sure if really needed)
        # audiopath = transpath.copy()
        # audiopath.update(datatype="audio", suffix=None, ext=".wav")
        # lastoffset = get_duration(path=audiopath)
        # uttdf.iloc[-1, list(uttdf.columns).index("offset")] = lastoffset

        # Transform transcript file into utterance records
        df = utterance2words(uttdf, nlp)
        df.to_csv(transpath.update(suffix="word", ext=".csv"), index=False)

        # Convert to textgrid
        try:
            tg = convert_wdf_tg(df)
            tg.save(
                transpath.update(suffix=None, ext=".TextGrid").fpath,
                format="long_textgrid",
                includeBlankSpaces=False,
                reportingMode="silence",
            )
        except Exception as e:
            print("[ERROR] converting file to TextGrid:", transfn, e)
            continue


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    args = parser.parse_args()
    main(args)


# This becomes a difficult problem:
# def validate_with_timinglog(uttdf: pd.DataFrame, transpath: Path):
#     conv, run = transpath["conv"], transpath["run"]

#     timepath = Path(root="stimuli", datatype="timing", suffix="events", ext=".csv")
#     timepath.update(conv=conv, run=run)

#     dft = pd.read_csv(timepath)

#     conv_id, run_id = int(conv), int(run)
#     trial_id = ((int(transpath["trial"]) - 1) % 4) + 1
#     dftrial = dft[(dft.run == run_id) & (dft.trial == trial_id)].copy()
#     dftrial["onset"] = dftrial["comm.time"]
#     dftrial["offset"] = dftrial["onset"].shift(-1)
#     dftrial = dftrial.iloc[1:-1]

#     # TODO - this is not exactly right
#     speaker, listener = conv_id, conv_id - 100
#     if transpath["first"] == "A":
#         speaker = conv_id - 100
#         listener = conv_id
#     dftrial["speaker"] = dftrial["role"].apply(
#         lambda x: speaker if x == "speaker" else listener
#     )
#     df = dftrial[
#         ["run", "trial", "item", "condition", "speaker", "onset", "offset"]
#     ].reset_index(drop=True)

#     if len(df) != len(uttdf):
#         print("[WARN] mismatch between transcript and timing log")
#     breakpoint()
