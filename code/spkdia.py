"""

convs=(1 2 ...)
for sub in $convs; do python code/spkdia.py -c $sub >> dia.log; done

one subject talked the entire trial:
conv-114_run-5_set-3_trial-17_item-17_condition-G_first-B_utterance.csv 
conv-128_run-1_set-1_trial-1_item-2_condition-G_first-A_utterance.csv

this had an utterance too short to give a speaker to, but seems correct:
conv-163_run-3_set-2_trial-9_item-9_condition-G_first-A_utterance.csv
"""
from glob import glob
from os import getenv

import pandas as pd
import whisperx
from constants import FNKEYS
from util.path import Path

HFTOK = getenv("HUGGING_FACE_HUB_TOKEN")


def diarize(df, audio, device: str = "cuda"):
    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HFTOK, device=device)

    breakpoint()
    df_segs = diarize_model(audio.fpath, min_speakers=2, max_speakers=2)

    df.insert(2, "offset", df.onset.shift(-1, fill_value=180))
    df.rename(
        columns={"onset": "start", "offset": "end", "speaker": "spk_sub"}, inplace=True
    )
    segments = list(df.to_dict(orient="index").values())

    result = whisperx.assign_word_speakers(df_segs, {"segments": segments})
    dfnew = pd.DataFrame.from_dict(result["segments"])
    dfnew["spk_id"] = dfnew.speaker.apply(lambda x: int(x[-2:]))

    corr = dfnew[["spk_sub", "spk_id"]].corr().iloc[0, 0]
    if corr != 1:
        # print('problem')
        breakpoint()
        raise RuntimeError(f"Speakers do not match: {corr}")


def main(args: dict):
    """Move and process transcripts."""

    # Look for transcripts
    transpath = Path(
        root="stimuli", datatype="transcript", suffix="utterance", ext=".csv"
    )
    transpath.update(**{str(k): v for k, v in vars(args).items() if k in FNKEYS})
    search_str = transpath.starstr(["conv", "datatype"])
    files = glob(search_str)
    files = sorted(files)
    assert len(files), "No files found for: " + search_str

    for transfn in files:
        transpath = Path.frompath(transfn)
        transpath.update(root="stimuli", datatype="transcript")

        uttdf = pd.read_csv(transfn)

        # Get audio file name
        audiopath = transpath.copy()
        audiopath.update(datatype="audio", suffix=None, ext=".wav")

        try:
            diarize(uttdf, audiopath)
        except Exception as e:
            print("Error at ", transpath, e)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    args = parser.parse_args()
    main(args)
