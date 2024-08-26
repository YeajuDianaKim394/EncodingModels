"""Project-level constants."""

TR = 1.5
SEED = 42

RUNS = (1, 2, 3, 4, 5)
TRIALS = (1, 2, 3, 4)
NTRIALS = len(TRIALS)
NRUNS = len(RUNS)

BLU = "#0173b2"  # HSL: (201, 99, 35)
ORG = "#de8f05"  # HSL: (38, 95, 45)
# from: print(sns.color_palette('colorblind').as_hex())

RUN_TRS = 544
TRIAL_TRS = 134  # excluding the first 12s (8 TR) blank on first trial of a run
CONV_TRS = 120

# Within each run, these are the indices of each trial, excluding prompt and blanks
RUN_TRIAL_SLICE = {
    1: slice(14, 134),
    2: slice(148, 268),
    3: slice(282, 402),
    4: slice(416, 536),
}

# filename keys, useful for ArgumentParser args object when updating Paths
FNKEYS = ("conv", "run", "trial")

# PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~"  # string.punctuation without [ ]
PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

EXCLUDED_CONVS = (
    101,  # acquisition error, missing TRs
    119,  # aborted scan
    168,  # excessive movement
    171,  # missing runs due to malfunction. aborted scan
)

# These conversations had interruptions midway through or had to be restarted
# `ls -1 sourcedata/CONV_scan/data/TimingsLog | cut -d_ -f2 | uniq -c | sort -nr``
# 103, 104, 108, 111, 116, 117, 121, 122, 127, 129, 138, 143, 154, 164, 167
INTERRUPTED_CONVS = (104, 108, 111, 116, 117, 122, 129, 138, 143)  # strangers

CONVS_STRANGERS = (
    104,
    105,
    106,
    107,
    108,
    111,
    112,
    114,
    116,
    117,
    120,
    122,
    123,
    126,
    128,
    129,
    131,
    132,
    133,
    137,
    138,
    142,
    143,
    153,
    156,
    157,
    158,
    163,
    174,
)
CONVS_FRIENDS = (
    103,
    109,
    113,
    118,
    121,
    127,
    130,
    134,
    145,
    146,
    147,
    148,
    150,
    151,
    152,
    154,
    155,
    159,
    160,
    161,
    162,
    164,
    165,
    166,
    167,
    169,
    170,
    172,
    173,
    175,
)
CONVS = CONVS_STRANGERS + CONVS_FRIENDS

SUBS_STRANGERS = list(CONVS_STRANGERS) + [c - 100 for c in CONVS_STRANGERS]
SUBS_FRIENDS = list(CONVS_FRIENDS) + [c - 100 for c in CONVS_FRIENDS]
SUBS = SUBS_FRIENDS + SUBS_STRANGERS

CONFOUND_REGRESSORS = [
    "a_comp_cor_00",
    "a_comp_cor_01",
    "a_comp_cor_02",
    "a_comp_cor_03",
    "a_comp_cor_04",
    "cosine00",
    "cosine01",
    "cosine02",
    "cosine03",
    "cosine04",
    "cosine05",
    "cosine06",
    "cosine07",
    "cosine08",
    "cosine09",
    "cosine10",
]

MOTION_CONFOUNDS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

EXTRA_MOTION_CONFOUNDS = [
    "trans_x_derivative1",
    "trans_y_derivative1",
    "trans_z_derivative1",
    "trans_x_derivative1_power2",
    "trans_y_derivative1_power2",
    "trans_z_derivative1_power2",
    "rot_x_derivative1",
    "rot_y_derivative1",
    "rot_z_derivative1",
    "rot_x_derivative1_power2",
    "rot_y_derivative1_power2",
    "rot_z_derivative1_power2",
    "trans_x_power2",
    "trans_y_power2",
    "trans_z_power2",
    "rot_x_power2",
    "rot_y_power2",
    "rot_z_power2",
]

ARPABET_PHONES = [
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
]
