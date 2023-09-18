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

    for conv in CONVS:
        # get the latest timing log
        files = sorted(glob(path.join(timingdir, f"CONV_{conv:03d}_TimingsLog*.csv")))
        if not len(files):
            print(f"[ERROR] no TimingLog exists for conversation {conv:03d}")
            continue

        dfs = []
        for filename in files:
            dfs.append(pd.read_csv(filename))
        df = pd.concat(dfs).reset_index(drop=True)

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
