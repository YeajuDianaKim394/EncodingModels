"""
to be run after running transcribe.sh
"""

import json
import os

import pandas as pd
from tqdm import tqdm
from util.path import Path

if __name__ == "__main__":
    files = os.listdir("stimuli/whisperx")

    for filename in tqdm(files):
        with open(os.path.join("stimuli", "whisperx", filename), "r") as f:
            d = json.load(f)
            df = pd.DataFrame(d["word_segments"])
            df.speaker.bfill(inplace=True)  # sometimes can be NaN for some reason

        path = Path.frompath(filename)
        path.update(root="stimuli", conv=path["conv"], datatype="whisperx", ext="csv")

        # restructure speaker column
        subB = int(path["conv"])
        subA = subB - 100
        first_speaker_id = subB
        second_speaker_id = subA
        if path["first"] == "A":
            first_speaker_id = subA
            second_speaker_id = subB

        # breakpoint()

        first_speaker = df.speaker.iloc[0]
        df["speaker"] = df["speaker"].apply(
            lambda x: first_speaker_id if x == first_speaker else second_speaker_id
        )

        column_to_move = df.pop("speaker")
        df.insert(0, "speaker", column_to_move)

        path.mkdirs()
        df.to_csv(path, index=False)
