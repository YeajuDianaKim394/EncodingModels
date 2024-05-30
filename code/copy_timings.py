"""Copy raw timing log files to BIDS.

Some sessions have multiple logs which need to be merged.
"""

from glob import glob
from os import path

import pandas as pd
from constants import CONVS, RUNS
from util.path import Path


def main():
    timingdir = "sourcedata/CONV_scan/data/TimingsLog"

    exceptions = [143]

    for conv in CONVS:
        if conv != 143:
            continue
        print(conv)
        # get the latest timing log
        files = sorted(glob(path.join(timingdir, f"CONV_{conv:03d}_TimingsLog*.csv")))
        if not len(files):
            print(f"[ERROR] no TimingLog exists for conversation {conv:03d}")
            continue

        filename = files[0]
        if len(files) > 1:
            if conv in exceptions:
                # load first, remove last run, and cat second file
                if len(files) > 2:
                    print("what do now?")

                dfs = []
                for j, filename in enumerate(files):
                    df = pd.read_csv(filename)
                    if j < len(files) - 1:
                        # remove last run
                        df = df[df.run != df.run.max()]
                    dfs.append(df)
                df = pd.concat(dfs).reset_index(drop=True)
            else:
                filename = files[-1]  # take latest
                df = pd.read_csv(filename)
                print("Choosing latest file from multiple", filename)
        else:
            df = pd.read_csv(files[0])

        newfile = Path(root="stimuli", datatype="timing", suffix="events", ext=".csv")
        conv_id = path.basename(files[-1]).split("_")[1]
        newfile.update(conv=conv_id)
        newfile.mkdirs()
        for run in RUNS:
            df2 = df[df.run == run].reset_index(drop=True)
            newfile.update(run=run)
            df2.to_csv(newfile, index=False)


if __name__ == "__main__":
    main()
