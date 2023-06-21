"""Move and normalizes raw transcripts into the stimuli folder.

# Pipeline
1. parse transcript file to determine each speaker and utterance
1. Normalize utterances:
"""

import re
from glob import glob
from os import path
from typing import Any

import pandas as pd
from util.path import Path

bracket_re = re.compile(r"[\[\(].*?[\]\)]")
inaudible_re = re.compile(r"\[inaudible.*\]")
laugh_re = re.compile(r"\([Ll]augh(s|ing|ter)?\)")
speaker_re = re.compile(r"\((\d{2}):(\d{2})\):$")
quote_re = re.compile(r"[’‘]")
space_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Clean up text

    maybe replacing inaudible with just [inaudible]
    and laughter with [laughter]

    english_us_arpa.dict
        <eps>   1.0     0.0     0.0     0.0     sil
        <unk>   0.99    0.6     2.02    0.8     spn
        [bracketed]     0.99    0.17    1.0     1.0     spn
        [laughter]      0.99    0.17    1.0     1.0     spn
    english_mfa.dict
        -d      0.99    0.24    1.0     1.0     spn
        [bracketed]     0.99    0.24    1.0     1.0     spn

    See non-speech-annotations at
    https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html
    """
    normalized_text = text

    # Remove weird apostraphe's
    normalized_text = quote_re.sub("'", normalized_text)

    # Replace inaudible with special token
    normalized_text = inaudible_re.sub("[inaudible]", normalized_text)

    # Replace laughts with special token
    normalized_text = laugh_re.sub("[laughter]", normalized_text)

    # # Replace remaining brackets
    # normalized_text = re.sub(bracket_re, "[bracketed]", normalized_text)

    # Remove double spaces
    normalized_text = space_re.sub(" ", normalized_text)

    # Strip any remaining edges
    normalized_text = normalized_text.strip()

    return normalized_text


def txt2csv(filepath: str | Path) -> pd.DataFrame:
    """Formats a plain-text transcript as a CSV.

    Example transcript:
    ```
        Speaker 1 (00:01):
        What do I value most in a friendship?...

        (00:51):
        But being funny is also always great...

        Speaker 2 (01:06):
        Um, I think w- when I think of a comfortable...

        (01:50):
        Um, I don't know, when I think of friendships...

        Speaker 1 (02:25):
        I definitely agree. I think, I guess maybe...
    ```
    and it has 3 turns and 5 utterances.

    """
    records = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line):
                if (match := speaker_re.search(line)) is not None:
                    speaker = None
                    if (start := match.start()) > 0:
                        speaker = line[:start].strip()
                    onset = int(match.group(1)) * 60 + int(match.group(2))
                    records.append([speaker, onset])
                else:
                    records[-1].append(line)

    df = pd.DataFrame(records, columns=("speaker", "onset", "text"))
    df.speaker.ffill(inplace=True)
    df["text"] = df["text"].apply(normalize_text)

    return df


def fwf2csv(filepath: str | Path) -> pd.DataFrame:
    """Fixed-width file to CSV. Unusued."""
    records = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line):
                speaker = line[:55].strip().strip(":")
                onset = line[55:105].strip()
                text = line[105:].strip()
                records.append((speaker, onset, text))
    df = pd.DataFrame(records, columns=("speaker", "onset", "text"))
    return df


def expand_columns(df: pd.DataFrame, conv: int, first: str) -> pd.DataFrame:
    """Infer speaker labels, and add offset and turn columns."""

    first_speaker = 0 if first == "A" else 1
    speakers = [conv - 100, conv]
    first_spk_label = df.speaker.iloc[0]
    mask = df.speaker == first_spk_label
    df.loc[mask, "speaker"] = speakers[first_speaker]
    df.loc[~mask, "speaker"] = speakers[(first_speaker + 1) % 2]

    df.insert(2, "offset", df.onset.shift(-1, fill_value=180))  # 180 s per trial
    df.reset_index(names="utterance", inplace=True)

    turns = (df.speaker.diff().abs().fillna(0).cumsum() / 100).astype(int)
    df.insert(0, "turn", turns)

    return df


def main(args):
    search_str = [""]
    if args.conv is not None:
        search_str.append(f"CONV_{args.conv:03d}")
    if args.run is not None:
        search_str.append(f"run_{args.run}")
    if args.trial is not None:
        search_str.append(f"trial_{args.trial}")
    if args.condition is not None:
        search_str.append(f"condition_{args.condition}")
    search_str.append(".txt")
    search_str = "*".join(search_str)

    transdir = "sourcedata/transcripts"

    files = glob(path.join(transdir, search_str))
    assert len(files), "No files found for: " + search_str

    for filepath in files:
        filename = path.splitext(path.basename(filepath.replace("CONV", "conv")))[0]
        entities: dict[str, str] = dict(zip(*[iter(filename.split("_"))] * 2))  # type: ignore

        df = txt2csv(filepath)
        df = expand_columns(df, conv=int(entities["conv"]), first=entities["first"])

        # Check if there's just one speaker
        if len(df.speaker.unique()) != 2:
            print("WARN less/more than two speakers", filename)

        # Save file
        transpath = Path(
            root="stimuli",
            datatype="transcript",
            suffix="utterance",
            ext="csv",
            **entities,
        )
        transpath.mkdirs()
        df.to_csv(transpath, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    parser.add_argument("--condition", type=str, default="G")
    args = parser.parse_args()
    main(args)
