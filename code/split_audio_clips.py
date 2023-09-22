"""
Splits audio clips and extracts generate trial audio
Author: LT
Modified: June 19, 2022

Modified by ZZ:
- change file paths and output name format
- add function to copy fixed wav files for transcripts with multiple log files (from Sebastian)
"""

# import libraries
import glob
import os
import re
import wave
from shutil import copy2

import pandas as pd
from util.path import Path

# set paths
path_base = os.getcwd()
path_data = os.path.join(path_base, "sourcedata", "CONV_scan", "data")
path_csv = os.path.join(path_data, "CONV_csv")
path_timingslog = os.path.join(path_data, "TimingsLog")
path_audio_raw = os.path.join(path_data, "RecordedAudio")
# path_audio_new = os.path.join(path_data, 'SegmentedAudio')
path_audio_new = os.path.join(path_base, "stimuli", "conv-{}", "audio")
print(path_audio_raw)
print(path_audio_new)

# create new dirs if they do not already exist
# if not os.path.exists(path_audio_new):
#     os.makedirs(path_audio_new)

# we are going to choose audio clips from prisma side so we don't transcribe duplicates
# prisma subj ids start with 1
subj_ids = [
    i[5:8]
    for i in list(
        filter(lambda v: re.match(f"^CONV_1[0-9][0-9]*", v), os.listdir(path_audio_raw))
    )
]

# default audio settings: mono, 16kHz sampling
channel = 1
rate = 16000


# wav creator function for the binary recorded & trimmed data
def wavmaker(raw_filepath, new_filepath, CHANNELS, RATE, start, end):
    # read raw audio
    f = open(raw_filepath, "rb")
    audio = f.read()
    # trim raw audio
    audio_clipped = audio[start:end]
    # create wav file
    wavef = wave.open(new_filepath, "w")
    wavef.setnchannels(CHANNELS)
    wavef.setsampwidth(2)
    wavef.setframerate(RATE)
    wavef.writeframes(audio_clipped)
    wavef.close()
    f.close()
    return


def original_code():
    # for each person, import csv info and audio clip
    for subj_name in subj_ids:
        try:
            # find csv
            subj_csv_filenames = list(
                filter(
                    lambda v: re.match(f"^CONV_{subj_name}.*.csv$", v),
                    os.listdir(path_csv),
                )
            )

            # if there's only one csv file for the subject, proceed
            if len(subj_csv_filenames) == 1:
                subj_csv_filepath = os.path.join(
                    path_csv,
                    list(
                        filter(
                            lambda v: re.match(f"^CONV_{subj_name}.*.csv$", v),
                            os.listdir(path_csv),
                        )
                    )[-1],
                )

                # extract information from csv
                subj_csv = pd.read_csv(subj_csv_filepath)

                # find timingslog
                subj_timingslog_filepath = os.path.join(
                    path_timingslog,
                    list(
                        filter(
                            lambda v: re.match(
                                f"^CONV_{subj_name}_TimingsLog.*.csv$", v
                            ),
                            os.listdir(path_timingslog),
                        )
                    )[-1],
                )
                # extract information from csv
                subj_timingslog = pd.read_csv(subj_timingslog_filepath)
                # find end audio position
                subj_timingslog_end_pos = subj_timingslog.iloc[-1]["audio_position"]
                # trim csv to only contain audio position for trial_intro
                subj_timingslog = subj_timingslog[
                    subj_timingslog["role"] == "trial_intro"
                ]

                # get trial-related info
                runs = subj_csv["run"]
                sets = subj_csv["set"]
                trials = range(1, len(subj_csv["run"]) + 1)
                items = subj_csv["item"]
                conditions = subj_csv["condition"]
                first_speakers = subj_csv["first_speaker"]
                audio_positions = subj_timingslog["audio_position"].reset_index(
                    drop=True
                )

                # find raw audio clip
                subj_audio_filepath = os.path.join(
                    path_audio_raw,
                    list(
                        filter(
                            lambda v: re.match(f"^CONV_{subj_name}.*RecordedAudio*", v),
                            os.listdir(path_audio_raw),
                        )
                    )[-1],
                )

                print(f"subject {subj_name}: {len(trials)} trials")

                for i in range(len(trials)):
                    # set new audio clip name
                    new_clip_dir = path_audio_new.format(subj_name)
                    os.makedirs(new_clip_dir, exist_ok=True)
                    new_clip_filepath = os.path.join(
                        new_clip_dir,
                        f"conv-{subj_name}_run-{runs[i]}_set-{sets[i]}_trial-{trials[i]}_item-{int(items[i])}_condition-{conditions[i]}_first-{first_speakers[i]}.wav",
                    )

                    # find start and end position for trial-specific audio segment
                    trial_start = audio_positions[i]
                    try:
                        trial_end = audio_positions[i + 1]
                    except:
                        # if end time is end position of the entire task
                        trial_end = subj_timingslog_end_pos

                    # clip raw audio and convert to wav
                    wavmaker(
                        subj_audio_filepath,
                        new_clip_filepath,
                        channel,
                        rate,
                        trial_start,
                        trial_end,
                    )

            else:
                print(f"There's more than 1 csv file for subject {subj_name}")
        except:
            print(f"Error - check # of trials for subject {subj_name}?")


def copy_fixes():
    wavfiles = glob.glob("sourcedata/audio_files/SegmentedAudio*/*.wav")
    print(f"Found {len(wavfiles)} files")

    new_root = os.path.join(path_base, "stimuli")
    for wavfn in wavfiles:
        # if '103' not in wavfn: continue  # if you want to run for a specific conversation
        basename = os.path.basename(wavfn).replace("CONV", "conv")
        string, _ = os.path.splitext(basename)
        parts = string.split("_")
        d = dict(zip(*[iter(parts)] * 2))
        d['item'] = str(int(float(d['item'])))

        wavpath = Path(**d, root=new_root, datatype="audio", ext=".wav")
        wavpath.mkdirs()

        if not wavpath.isfile():
            copy2(wavfn, wavpath)
        else:
            print("File already exists: ", wavpath)


if __name__ == "__main__":
    # original_code()
    copy_fixes()
