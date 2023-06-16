# brain-to-brain hyperscanned fMRI conversations


naming convention of conversations: conv-xxx is for subject 02 and 102
subjects xx and 1xx are a dyad.

## important files

- `CONV/data/CONV_Postscan_Global.csv`  -- friend/stranger condition
- `/jukebox/tamir/Sebastian/3Dmodel/LanguageDecoding/raw_transcripts_from_Revs`


## directory structure

```bash
.
├── code
│   ├── util
│   └── *.py  # see below for full description
├── data                
│   ├── audio           
│   │   ├── aligned     
│   │   └── segmented   
├── data/source
│   ├── CONV  # all fMRI data and fMRIPrep preprocessed:
│   │   ├── data/bids/derivatives/fmriprep
│   ├── CONV_scan  # see below
│   │   └── 
│   └── 
└── 
```
Generate with `tree -d -L 3`

Behavrioal data is in `data/source/Conv_scan/data`:

```bash
.
|- black_log/ # psychopy logs for hyperalignment task (Black)
|- CONV_csv/ # csv experiment logs for CONV task - important!
|- CONV_log/ # psychopy logs for CONV task
|- CONV_psydat/ # psychopy logs for CONV task
|- RecordedAudio/ # raw audio files for CONV task
|- SegmentedAudio/ # work in progress - segmented .wav files from RecordedAudio (trial-specific audio)
|- timestamps/ # client & server timestamps
|- TimingsLog/ # contains audio position for each turn - important!
|- TTL_timestamps/ # other timing-related logs
|- stimuli/ # contains stimuli scripts for the experiment
|- misc/ # some folders can be ignored
|- reorganize_data.py # moves files dumped into the data directory into their corresponding subdirectories
|- count_discrepancies_files.py # checks and sees if partners in a dyad have different # of files. If so, flag as something to check
|- discrepancies.txt # file created when saving the output of count_discrepancies_files.py
|- split_audio_clips.py # work in progress - segments raw audio clips into .wav clips for each trial
```

## code

### transcript preprocessing
We have transcripts and audio at the trial level. Each transcript contains utterances per speaker turn and the utterance onset.

1. `split_audio_clips.py` splits one long audio file into per run/trial files
1. `copy_transcripts.py` copies raw transcripts as-is into our folder structure
1. `fix_transcripts.sh` fix transcription problems
1. `process_transcripts.py` pipeline to normalize, sentencize, and tokenize transcripts and prepare for alignment
1. `align_transcript.sh` run forced-alignment to get word-level onset/offsets
1. `merge_transcripts.py` merge our utterance-level transcripts with forced aligner word-level
1. `descriptive_stats.ipynb` for QA

files:
- trial-level .wav
- trial-level raw transcript (.txt)
- trial-level cleaned transcript (.csv)
- trial-level transcript TextGrid (.TextGrid)
- trial-level aligned transcript TextGrid (_aligned.TextGrid)
- trial-level aligned csv (_aligned.csv)

#### useful commands

count how many transcripts each conversation has
```
ls -1 sourcedata/raw_transcripts_from_Revs/ | cut -d '_' -f 2 | sort | uniq -c | less -N
ls -1 data/audio_files/SegmentedAudio_Seb4 | cut -d '_' -f 2 | sort | uniq -c | less -N
```

find specific regex in transcripts
```
# Find numbers
grep -norE ' [0-9]+:[0-9]+ ' sourcedata/raw_transcripts_from_Revs
grep -norE '[0-9]+,[0-9]+' sourcedata/raw_transcripts_from_Revs

# Find words with &
grep -ohrE '.\&.' sourcedata/raw_transcripts_from_Revs

# Find bracketed items
grep -ohrE '\[.*\]' sourcedata/raw_transcripts_from_Revs
grep -ohrE '\(\w*\)'sourcedata/raw_transcripts_from_Revs | sort | uniq -c | sort -rh
```

Convert aligned csv to audacity labels to verify alignment:
```
function t2aud() { awk -v OFS="\t" -F"," '{print $10,$11,$8}' "$1" | tail -n +2 }
```