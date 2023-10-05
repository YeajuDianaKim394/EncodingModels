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

# PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~"  # string.punctuation without [ ]
PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

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

SUBS_STRANGERS = list(CONVS_STRANGERS) + [c - 100 for c in CONVS_STRANGERS]
SUBS_FRIENDS = list(CONVS_FRIENDS) + [c - 100 for c in CONVS_FRIENDS]
SUBS = SUBS_FRIENDS + SUBS_STRANGERS

CONFOUNDS = [
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
    "trans_x",
    "trans_x_derivative1",
    "trans_x_derivative1_power2",
    "trans_x_power2",
    "trans_y",
    "trans_y_derivative1",
    "trans_y_derivative1_power2",
    "trans_y_power2",
    "trans_z",
    "trans_z_derivative1",
    "trans_z_derivative1_power2",
    "trans_z_power2",
    "rot_x",
    "rot_x_derivative1",
    "rot_x_derivative1_power2",
    "rot_x_power2",
    "rot_y",
    "rot_y_derivative1",
    "rot_y_derivative1_power2",
    "rot_y_power2",
    "rot_z",
    "rot_z_derivative1",
    "rot_z_power2",
    "rot_z_derivative1_power2",
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
