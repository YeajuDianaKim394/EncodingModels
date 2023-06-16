"""Move raw transcripts into our stimuli folder.
"""

import re
from glob import glob
from os import path

import pandas as pd
from util.path import Path


def fwf2csv(filepath: str | Path) -> pd.DataFrame:
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


speaker_pattern = re.compile(r"\((\d{2}):(\d{2})\):$")

def txt2csv(filepath: str | Path) -> pd.DataFrame:
    records = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line):
                if (match := speaker_pattern.search(line)) is not None:
                    speaker = None
                    if (start := match.start()) > 0:
                        speaker = line[:start].strip()
                    onset = int(match.group(1)) * 60 + int(match.group(2))
                    records.append([speaker, onset])
                else:
                    records[-1].append(line)

    df = pd.DataFrame(records, columns=("speaker", "onset", "text"))
    df.speaker.ffill(inplace=True)
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

    for filename in files:
        # df = fwf2csv(filename)
        df = txt2csv(filename)

        basename = path.splitext(path.basename(filename.replace("CONV", "conv")))[0]
        fileparts = basename.split("_")
        fileparts = dict(zip(*[iter(fileparts)] * 2))
        newfn = Path(
            **fileparts,
            root="stimuli",
            datatype="transcript",
            suffix="utterance",
            ext="csv",
        )
        newfn.mkdirs()
        df.to_csv(newfn, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    parser.add_argument("--condition", type=str, default="G")
    args = parser.parse_args()
    main(args)
