"""forced-alignment of existing transcripts with whisperx

conv-103_run-1_set-1_trial-2_item-2_condition-G_first-B_utterance.csv            
Failed to align segment ("Um, uh I'd want to be famous for something, I like to think that would be pretty cool, if I could just like, for that."): backtrack failed, resorting to original...
Failed to align segment ("But honestly, I, like if you make a big discovery in science of some sort, um if you invent something. I don't know, that'd be pretty cool. Sure, I could do it."): backtrack failed, resorting to original...
Failed to align segment ("Um I think I want to be famous also, probably for something with science. But I think that if I we re to be famous for something like, like a celebrity like the Kardashians or something, I think that'd be overwhelming."): b acktrack failed, resorting to original...

conv-103_run-3_set-1_trial-6_item-9_condition-G_first-B_utterance.csv  
Failed to align segment ("It's funny how you said, um, like when. I guess, you just would take, would take the stuff to bed and be like, "Oh, it's like the could then it won't happen."): backtrack failed, resorting to original...

conv-103_run-5_set-3_trial-16_item-19_condition-G_first-B_utterance.csv
Failed to align segment ("How many times have you guys moved before? Well, once, twice?"): backtrack failed, resorting to original...

Error at  conv-109_run-2_set-1_trial-7_item-7_condition-G_first-B_utterance.csv
object of type 'float' has no len()

Failed to align segment ("Um, yeah. No, skydiving is really cool and I think it'd be awesome to get to do it. Um, I think I've, uh, I've been interested in like, obviously, like getting to drive a fast car or something. That would be, I feel like that would be fun. That's something that I've wanted to do. Um, I mean, obviously, costs kind of a prohibitive factor on that and also safety. Like, yeah."): backtrack failed, resorting to original...
Error at  conv-150_run-3_set-2_trial-9_item-10_condition-G_first-A_utterance.csv
Something wrong with alignment. Double check

Error at  conv-155_run-5_set-3_trial-19_item-20_condition-G_first-A_utterance.csv
object of type 'float' has no len()
Failed to align segment ("No, yeah, definitely. But, like, you gotta think about, like, the UV rays from the sun, like, that's, like, called, like, UV radiation. You gotta think about, like, cancer. Y- you don't want skin cancer."): backtrack failed, resorting to original...

Error at  conv-165_run-2_set-1_trial-5_item-6_condition-G_first-B_utterance.csv
Something wrong with alignment. Double check

Error at  conv-172_run-3_set-2_trial-10_item-10_condition-G_first-A_utterance.csv
object of type 'float' has no len()

"""

from glob import glob

import pandas as pd
import whisperx
from constants import FNKEYS
from util.path import Path


def word_align(
    segments: list, audio_file: Path | str, device: str = "cuda", language: str = "en"
):
    audio = whisperx.load_audio(audio_file)
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        segments, model_a, metadata, audio, device, return_char_alignments=False
    )

    return result


def align(uttdf: pd.DataFrame, audio_file: Path | str) -> pd.DataFrame:
    # Reformat into segments
    uttdf.rename(columns={"onset": "start", "offset": "end"}, inplace=True)
    segments = list(uttdf.to_dict(orient="index").values())

    result = word_align(segments, audio_file)
    new_segments = result["segments"]

    if len(new_segments) != sum(len(s["sentence_spans"]) for s in segments):
        raise ValueError("Something wrong with alignment. Double check")

    dfs = []
    i = 0
    for segment in segments:
        for j, sentspan in enumerate(segment["sentence_spans"], start=1):
            if i >= len(new_segments):
                print("wah")
                breakpoint()
            df = pd.DataFrame(new_segments[i]["words"])
            df.insert(0, "sentence", j)
            df.insert(0, "speaker", segment["speaker"])
            i += 1
            dfs.append(df)

    if i != len(new_segments):
        print("something wrong, not same number of segments as sents")
        breakpoint()

    dfn = pd.concat(dfs)
    return dfn


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
        # Fill nan to 180 s per trial
        uttdf.insert(2, "offset", uttdf.onset.shift(-1, fill_value=180))

        # Get audio file name
        audiopath = transpath.copy()
        audiopath.update(datatype="audio", suffix=None, ext=".wav")

        try:
            dfn = align(uttdf, audiopath)
            newpath = transpath.update(suffix="aligned", ext=".csv")
            dfn.to_csv(newpath, index=False)
        except Exception as e:
            print("Error at ", transpath)
            print(e)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    args = parser.parse_args()
    main(args)
