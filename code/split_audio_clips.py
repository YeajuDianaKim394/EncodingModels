"""
Splits audio clips and extracts generate trial audio
rewritten by ZZ 
"""

import wave
from glob import glob
from os import path

import pandas as pd
from constants import CONVS_STRANGERS
from tqdm import tqdm
from util.path import Path

# from whisperx.transcribe import load_audio


def main():

    path_data = path.join("sourcedata", "CONV_scan", "data")
    path_csv = path.join(path_data, "CONV_csv")
    path_timingslog = path.join(path_data, "TimingsLog")
    path_audio_raw = path.join(path_data, "RecordedAudio")

    for conv in tqdm(CONVS_STRANGERS):
        csv_files = glob(path.join(path_csv, f"CONV_{conv}*.csv"))
        timing_files = glob(path.join(path_timingslog, f"CONV_{conv}_TimingsLog*.csv"))
        audio_file = glob(path.join(path_audio_raw, f"CONV_{conv}_RecordedAudio*"))

        if not len(csv_files):
            raise FileNotFoundError("No CSV file found")

        if not len(timing_files):
            raise FileNotFoundError("No Timing file found")

        if not len(audio_file):
            raise FileNotFoundError("No Audio File file found")

        subj_df, timing_df = None, None

        # CSV files
        if len(csv_files) > 1:
            try:
                dfs = [pd.read_csv(filename) for filename in sorted(csv_files)]
            except Exception:
                print(csv_files)
                raise ValueError("bad")

            # combine
            if conv in [122, 129]:
                subj_df = dfs[-1]
            elif conv in [108, 111, 116, 117]:
                subj_df = pd.concat(dfs, ignore_index=True, axis=0)
                subj_df.sort_values(["run", "trial"], inplace=True)
                subj_df.reset_index(inplace=True, drop=True)
            elif conv == 143:
                dfs[0] = dfs[0][dfs[0]["run"] < 3]
                subj_df = pd.concat(dfs, ignore_index=True, axis=0)
            else:
                print(f"MORE THAN ONE CSV {conv}", len(csv_files))
                breakpoint()

        # Timing files
        if len(timing_files) > 1:
            dfs = [pd.read_csv(filename) for filename in sorted(timing_files)]

            # take the last one
            if conv in [104, 122, 129, 138]:
                timing_df = dfs[-1]
            elif conv in [108, 111, 116, 117]:
                dfs[1]["audio_position"] += dfs[0]["audio_position"].iloc[-1]
                timing_df = pd.concat(dfs, ignore_index=True, axis=0)
            elif conv == 143:
                dfs[0] = dfs[0][dfs[0]["run"] < 3]
                dfs[1]["audio_position"] += dfs[0]["audio_position"].iloc[-1]
                timing_df = pd.concat(dfs, ignore_index=True, axis=0)
            else:
                print(f"MORE THAN ONE TIMING {conv}", len(timing_files))
                breakpoint()

        # extract information from csv
        timing_log = pd.read_csv(timing_files[-1]) if timing_df is None else timing_df

        outpath = Path(
            root="data/stimuli",
            datatype="timing",
            conv=conv,
            suffix="events",
            ext=".csv",
        )
        outpath.mkdirs()
        timing_log.to_csv(outpath.fpath, index=False)

        # find end audio position
        subj_timingslog_end_pos = timing_log.iloc[-1]["audio_position"]
        # trim csv to only contain audio position for trial_intro
        timing_log = timing_log[timing_log["role"] == "trial_intro"]
        audio_positions = timing_log["audio_position"].reset_index(
            drop=True
        ).tolist() + [subj_timingslog_end_pos]

        # get trial-related meta-data
        subj_csv = pd.read_csv(csv_files[-1]) if subj_df is None else subj_df
        runs = subj_csv["run"]
        sets = subj_csv["set"]
        trials = subj_csv["trial"]
        items = subj_csv["item"]
        conditions = subj_csv["condition"]
        first_speakers = subj_csv["first_speaker"]

        with open(audio_file[0], "rb") as f:
            audio = f.read()

        for i in range(len(subj_csv)):

            # find start and end position for trial-specific audio segment
            trial_start = audio_positions[i]
            trial_end = audio_positions[i + 1]

            # clip raw audio and convert to wav
            audio_clipped = audio[trial_start:trial_end]

            outpath = Path(
                root="data/stimuli",
                datatype="audio",
                conv=conv,
                run=runs[i],
                trial=trials[i],
                set=sets[i],
                item=items[i],
                condition=conditions[i],
                first=first_speakers[i],
                ext=".wav",
            )
            outpath.mkdirs()

            with wave.open(outpath.fpath, "wb") as wavef:
                wavef.setnchannels(1)
                wavef.setsampwidth(2)
                wavef.setframerate(16000)
                wavef.writeframes(audio_clipped)

            # audio = load_audio(outpath.fpath)
            # print(outpath.fpath, audio.shape)


if __name__ == "__main__":
    main()
