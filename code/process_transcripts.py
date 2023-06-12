"""Move and modify transcripts to prepare for alignment with audio.

Creates a csv and TextGrid file version of each transcript.

Pipline
-------

TODO:  replace inaudible with silences?

1. parse transcript file to determine each speaker and utterance
1. Normalize utterances:
    0. remove brackets
    0. remove multiple white space
    0. remove whitespace before punctuation  (do a grep for these instances)
    0. TODO replace numbers with text version?
1. Tokenize utterances and segment into sentences
    0. TODO todo consider using vocab from MFA dictionary?
1. Normalize each token
    0. remove punctuation, white space, and lower case
1. Save CSV with the following columns:
    - speaker, utt_id, utt_onset, utt_offset, sent_id, token_id, word, word_normalized
    0. ensures that joining all the words will return the original (cleaned) text
    0. this is the file we will join with the output of forced alignment
1. Create a TextGrid with an Tier per speaker using the joined word_normalized column per utt_id group

https://github.com/facebookresearch/fairseq/blob/main/examples/mms/data_prep/text_normalization.py
https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/montreal_forced_aligner/corpus/multiprocessing.py#L260
https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html#text-normalization-and-dictionary-lookup

    - consider replacing numbers with the actual speech so we align better.
"""

import re
import string
from glob import glob
from os import path

import pandas as pd
import spacy
from librosa import get_duration
from util.path import Path
from util.transcription import records2tg

transdir = "sourcedata/raw_transcripts_from_Revs"
timingdir = "sourcedata/CONV_scan/data/TimingsLog"
stimdir = "stimuli"

# Speaker pattern
speaker_pattern = re.compile(r"\((\d{2}):(\d{2})\):$")
bracket_pattern = re.compile(r"[\[\(].*?[\]\)]")

# These things break MFA
custom_substitutions = {101: [("somethin'", "something")]}


def clean_utterance(text: str) -> str:
    """Clean up text"""
    normalized_text = re.sub(bracket_pattern, "", text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return normalized_text


def transcript2records(filepath: str, conv: int, first: str) -> list:
    """Function to process transcript into standard format.

    Returns a list of dictionaries each with entries:
    speaker, onset, offset, and text.
    """
    records = []
    turn = 0
    first_speaker = 0 if first == "A" else 1
    speakers = [conv - 100, conv]
    with open(filepath, "r") as f:
        entry = {}
        for line in f.readlines():
            line = line.strip()
            if len(line):  # skip empty lines
                if (match := speaker_pattern.search(line)) is not None:
                    minutes = int(match.group(1))
                    seconds = int(match.group(2))
                    entry["onset"] = minutes * 60 + seconds
                    entry["offset"] = 0

                    # Only switch turns if speakers actually change
                    if line[0] != "(":
                        first_speaker += 1
                        turn += 1
                    entry["speaker"] = speakers[first_speaker % 2]
                    entry["turn"] = turn
                else:
                    entry["text"] = line
                    entry["utterance"] = len(records)
                    records.append(entry)
                    entry = {}

        # Fill offsets
        for i in range(1, len(records)):
            entry_n = records[i]
            entry_nm1 = records[i - 1]
            entry_nm1["offset"] = entry_n["onset"]

    return records


def transcript2turns(filepath: str) -> list:
    with open(filepath, 'r') as f:
        turns = []
        for line in f.readlines():
            line = line.strip()
            if len(line):  # skip empty lines
                if speaker_pattern.search(line) is None:
                    turns.append(line)
    return turns


def main(args):
    """Move and process transcripts."""

    transpath = Path(root='stimuli', datatype='audio', suffix='transcript', ext='.txt')
    transpath.update(**vars(args))
    search_str = transpath.starstr(["conv", "datatype"])
    print(search_str)

    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    # TODO https://pypi.org/project/inflect/
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
    suffixes = list(nlp.Defaults.suffixes)
    suffixes.remove("'s")
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search

    for transfn in files:

        transpath = Path.frompath(transfn)
        transpath.update(root='stimuli', datatype='audio')
        conv_id = transpath['conv']
        run_id = transpath['run']
        trial_id = ((int(transpath["trial"]) - 1) % 4) + 1

        # Find and read timing log file
        timingfn = sorted(
            glob(path.join(timingdir, f"CONV_{conv_id:03d}_TimingsLog_*.csv"))
        )[-1]
        dft = pd.read_csv(timingfn)
        newfn = transpath.copy()
        del newfn["trial"], newfn["set"], newfn["item"], newfn["first"], newfn["condition"]
        newfn.update(suffix='events', ext='.csv')
        dft[dft.run == run_id].reset_index().to_csv(newfn, index=False)
        print(newfn)
        continue

        # Choose a strategy to align utterances
        use_timinglog = not True
        if use_timinglog:
            dftrial = dft[(dft.run == run_id) & (dft.trial == trial_id)].copy()
            dftrial['onset'] = dftrial['comm.time']
            dftrial['offset'] = dftrial['onset'].shift(-1)
            dftrial = dftrial.iloc[1:-1]

            speaker, listener = conv_id, conv_id - 100
            if transpath['first'] == 'A':
                speaker = conv_id - 100
                listener = conv_id
            dftrial['speaker'] = dftrial['role'].apply(lambda x: speaker if x == 'speaker' else listener)
            df = dftrial[['run', 'trial', 'item', 'condition', 'speaker', 'onset', 'offset']].reset_index(drop=True)
            utterances = transcript2turns(transfn)

            records = []
            for i, text, (idx, record) in zip(range(len(utterances)), utterances, df.to_dict(orient='index').items()):
                record['turn'] = idx
                record['text'] = text
                record['utterance'] = i
                records.append(record)
        else:
            # Transform transcript file into utterance records
            try:
                records = transcript2records(
                    transfn, conv=conv_id, first=transpath["first"]
                )
                transpath.update(suffix=None, ext='.wav')
                records[-1]["offset"] = get_duration(path=transpath)
            except KeyError as e:
                print("[ERROR] failed to parse", transfn, e)
                continue

        # Clean up utterances
        for entry in records:
            text = entry["text"]
            if substitutions := custom_substitutions.get(conv_id):
                for subs in substitutions:
                    text = text.replace(subs[0], subs[1])
            text = clean_utterance(entry["text"])
            entry["text"] = text

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

        # Aggregate into DataFrame
        df = pd.DataFrame(records)
        df = df.explode(["sentence_id", "is_punct", "token"], ignore_index=True)
        df.drop("text", axis=1, inplace=True)
        df["is_punct"] = df.is_punct.astype(bool)
        df["token_norm"] = (
            df.token.str.strip().str.lower().str.strip(string.punctuation)
        )
        # sents = df.groupby(['speaker', 'turn', 'sentence_id']).token.apply(''.join)

        # Create TextGrid
        turns = (
            df[~df.is_punct]
            .groupby("utterance")
            .agg(
                {
                    "speaker": "first",
                    "token_norm": " ".join,
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
        try:
            tg = records2tg(utt_records)
        except Exception as e:
            print("[ERROR] converting file to TextGrid:", transfn, e)
            # print(turns)
            continue

        # Save textgrid
        tg.save(
            transpath.update(ext=".TextGrid"),
            format="long_textgrid",
            includeBlankSpaces=False,
            reportingMode="silence",
        )
        # Save transcript
        df.to_csv(transpath.update(suffix="transcript", ext=".csv"), index=False)

        # Save run timing file ( will be overwritten each turn but ok )
        # del newfn["trial"], newfn["set"], newfn["item"], newfn["first"]
        # newfn.update(suffix="timing", ext=".csv")
        # dft[dft.run == run_id].reset_index().to_csv(newfn)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    args = parser.parse_args()
    main(args)

