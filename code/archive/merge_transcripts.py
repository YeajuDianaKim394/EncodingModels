"""Align aligned words with original transcript

"""

from glob import glob

import pandas as pd
from constants import FNKEYS
from praatio import textgrid
from util.path import Path

# ideally we can tokenize these in spacy instead
tokenmerge = {
    "M&M's": ('m', "m's"),
    "2,000": ('2', '000'),
    "9:30": ('9', '30'),
    "U.S.": ('u', 's'),
    "4.0": ('4', '0'),
    "R&B": ('r', 'b'),
    "20,000": ('20', '000'),
    "M-684": ('m', '684'),
    "a.m.": ('a', 'm'),
    "5:00": ('5', '00'),
    "0:05": ('0', '05'),
    "6:00": ('6', '00'),
    "9:00": ('9', '00'),
    "5:30": ('5', '30'),
    "10:00": ('10', '00'),
    "11:00": ('11', '00'),
    "12:00": ('12', '00'),
    "4:00": ('4', '00'),
    "6:30": ('6', '30'),
    "7:00": ('7', '00'),
    "L.A.": ('l', 'a'),
}

def postfix(word_df: pd.DataFrame, utt_df: pd.DataFrame, inplace:bool=True) -> pd.DataFrame:
    """

    When there's a mismatch, it's usually because MFA split a token more than we expected,
    so we need to find those instances and merge those rows in word_df before aligning.
    """
    rm_ids = []
    for word, parts in tokenmerge.items():
        if (utt_df.token.str.strip() == word).any():
            part_ids = (word_df.token == parts[0]).values.nonzero()[0]
            for part_id in part_ids:
                if word_df.token.iloc[part_id+1] == parts[1]:
                    print('Replacing', word, parts, part_id)
                    word_df.loc[part_id, 'token'] = word.lower()
                    word_df.loc[part_id, 'offset'] = word_df.offset.iloc[part_id+1]
                    rm_ids.append(part_id + 1)
    word_df.drop(rm_ids, inplace=inplace)
    word_df.reset_index(drop=True, inplace=inplace)
    return word_df



def merge_transcripts(
    utterance_fn: str | Path, words_fn: str, interactive: bool = True
) -> pd.DataFrame:
    # Read word transcript
    dfs = []
    tg = textgrid.openTextgrid(words_fn, includeEmptyIntervals=False)
    for tier in tg.tiers:
        if "words" in tier.name:
            words = [(s, e, w.strip()) for s, e, w in tier.entries]
            df = pd.DataFrame(words, columns=["onset", "offset", "token"])
            speaker = tier.name.split(" ")[0]
            if speaker.isnumeric():
                speaker = int(speaker)
            # TODO - can infer this instead of doing it programatically
            # these are exceptions where there was only one speaker for the trial
            elif utterance_fn.basename.startswith('conv-114_run-5_set-3_trial-17'):
                speaker = 114
            elif utterance_fn.basename.startswith('conv-128_run-1_set-1_trial-1'):
                speaker = 128
            elif utterance_fn.basename.startswith('conv-143_run-4_set-3_trial-8'):
                speaker = 143
            elif utterance_fn.basename.startswith('conv-153_run-5_set-3_trial-18'):
                speaker = 53
            else:
                breakpoint()
                raise ValueError(f"Unknown speaker {speaker}")
            df.insert(0, "subject", speaker)
            dfs.append(df)
    word_df = pd.concat(dfs)
    word_df.sort_values("onset", ascending=True, inplace=True, ignore_index=True)

    # is this needed?
    if len(word_df.subject.unique()) < 2:
        message = "Only one subject aligned"
        print(message)
        # if interactive:
        #     print(message)
        #     breakpoint()
        # if not utterance_fn.basename.startswith('conv-114_run-5_set-3_trial-17'):
        #     raise ValueError(message)

    # Read utterance transcript
    utt_df = pd.read_csv(utterance_fn, index_col=None)
    utt_df = utt_df[~utt_df.is_punct].reset_index(drop=True)
    utt_df.drop(["onset", "offset", "is_punct"], axis=1, inplace=True)

    if len(utt_df) != len(word_df):
        word_df = postfix(word_df, utt_df)

    # utt_df = utt_df.drop(337).reset_index(drop=True) # only for -c 120 -r 1 -t 3
    # utt_df = utt_df.drop(53).reset_index(drop=True) # only for -c 157 -r 3 -t 10
    # utt_df = utt_df.drop(121).reset_index(drop=True) # conv-156_run-1_set-1_trial-4 "$8"
    # utt_df = utt_df.drop(53).reset_index(drop=True) # conv-157_run-3_set-2_trial-10  "$20"

    # Now merge them
    if len(utt_df) == len(word_df):
        new_df = pd.concat((utt_df, word_df[["onset", "offset"]]), axis=1)
        unmatching = utt_df.token_norm.str.lower() != word_df.token.str.lower()
        if unmatching.any():
            message = "Matching lengths but mis-matching words"
            if interactive:
                print(message)
                print(new_df[unmatching])
                breakpoint()
            # raise ValueError(message)
    else:
        new_df = pd.concat((utt_df, word_df[["onset", "offset", "token"]]), axis=1)
        message = f"Mis-matching lengths: {len(utt_df)} in utt, {len(word_df)} in words"
        if interactive:
            print(message)

            # useful when the difference is small
            ids = (
                new_df.token_norm.str.lower() != new_df.token.iloc[:, 1]
            ).values.nonzero()
            print(ids[0][0])
            print(new_df.iloc[ids[0][0]-5:ids[0][0]+5])

            # useful when the difference is big (e.g. missing an utterance)
            wordutts = word_df.groupby('subject').token.apply(lambda x: ' '.join(x))
            print(wordutts)

            breakpoint()
        raise ValueError(message)

    return new_df


def main(args):
    """Move and process transcripts."""

    transpath = Path(root="stimuli", datatype="aligned", ext=".TextGrid")
    transpath.update(**{k: v for k, v in vars(args).items() if k in FNKEYS})
    search_str = transpath.starstr(["conv", "datatype"])

    files = glob(search_str)
    assert len(files), "No files found for: " + search_str

    for tgpath in files:
        # look for matching csv
        csvpath = Path.frompath(tgpath)
        csvpath.update(
            root="stimuli", datatype="transcript", suffix="word", ext=".csv"
        )

        if csvpath.isfile():
            try:
                df = merge_transcripts(csvpath, tgpath, interactive=args.interactive)
                if df is not None:
                    df.insert(0, "trial", csvpath["trial"])
                    df.insert(0, "run", csvpath["run"])
                    new_fn = csvpath.update(datatype="aligned", ext=".csv")
                    df.to_csv(new_fn)
            except Exception as e:
                print("[ERROR]", e, csvpath)
                continue
        else:
            print("[ERROR] no TextGrid found for:", csvpath.basename)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--conv", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-t", "--trial", type=int)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    main(args)


# PUNC = '\'"`、。।，@<>"(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=‘'  # from MFA
# PUNC_SET = set(PUNC)
# def split_turn(text: str) -> list:
#     words = text.split()  # [sanitize(x) for x in text.split()]
#     words = [word.split('-') for word in words]  # MFA treates hyphenated words separately
#     # words = [x for x in words if x not in ['', '-', "'"]]
#     words = [word for word in sum(words, []) if word != '' and word not in PUNC_SET]
#     return words
