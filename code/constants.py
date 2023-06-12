"""Project-level constants."""

TR = 1.5
SEED = 42

RUNS = (1, 2, 3, 4, 5)
TRIALS = (1, 2, 3, 4)
NTRIALS = len(TRIALS)
NRUNS = len(RUNS)

# filename keys, useful for ArgumentParser args object when updating Paths
FNKEYS = ('conv', 'run', 'trial')

CONVS_STRANGERS = (
    101,
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
    119,
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
    171,
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
