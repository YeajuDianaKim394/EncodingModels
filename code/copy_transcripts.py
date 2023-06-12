"""Move raw transcripts into our stimuli folder.
"""
from glob import glob
from os import path
from shutil import copy

from util.path import Path


def main(args):

    search_str = [""]
    if args.conv is not None:
        search_str.append(f"CONV_{args.conv:03d}")
    if args.run is not None:
        search_str.append(f"run_{args.run}")
    if args.trial is not None:
        search_str.append(f"trial_{args.trial}")
    search_str.append(".txt")
    search_str = "*".join(search_str)

    transdir = "sourcedata/raw_transcripts_from_Revs"
    files = glob(path.join(transdir, search_str))
    assert len(files), "No files found for: " + search_str

    for filename in files:
        basename = path.splitext(path.basename(filename.replace("CONV", "conv")))[0]
        fileparts = basename.split("_")
        fileparts = dict(zip(*[iter(fileparts)] * 2))
        newfn = Path(
            **fileparts,
            root="stimuli",
            suffix="transcript",
            ext="txt",
            datatype="audio",
        )
        newfn.mkdirs()
        copy(filename, newfn)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    args = parser.parse_args()
    main(args)
