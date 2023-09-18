"""Project-level constants."""

TR = 1.5
SEED = 42

RUNS = (1, 2, 3, 4, 5)
TRIALS = (1, 2, 3, 4)
NTRIALS = len(TRIALS)
NRUNS = len(RUNS)

RUN_TRS = 544
TRIAL_TRS = 134  # excluding the first 12s (8 TR) blank on first trial of a run
CONV_TRS = 120

# filename keys, useful for ArgumentParser args object when updating Paths
FNKEYS = ("conv", "run", "trial")

PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~"  # string.punctuation without [ ]

EXCLUDED_CONVS = (
    101,  # acquisition error, missing TRs
    119,  # aborted scan
    168,  # excessive movement
    171,  # missing runs due to malfunction. aborted scan
)

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

CONFOUNDS = [
    "a_comp_cor_00",
    "a_comp_cor_01",
    "a_comp_cor_02",
    "a_comp_cor_03",
    "a_comp_cor_04",
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "cosine00",
]
