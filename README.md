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
│   └── *.py                     # see below for full description
├── data                
│   ├── sub-*
│   └── derivatives/fmriprep     # preprocessed BOLD data
├── stimuli
│   ├── conv-*
│   │   ├── audio/
│   │   │   └── *.wav            # per-trial audio file
│   │   ├── transcript/
│   │   │   ├── *_utterance.csv  # cleaned up original transcript
│   │   │   ├── *_word.csv       # tokenized, sentencized word-level transcript
│   │   │   └── *.TextGrid       # utterance-level transcript based on tokens
│   │   ├── aligned/
│   │   │   ├── *.TextGrid       # output of forced aligner
│   │   │   └── *_word.csv       # aligned textgrid merged with word-level transcript
│   │   ├── timing/
│   │   └── └── *_events.csv     # timinglog
│   └── 
└── 
```

Behavrioal data is in `sourcedata/Conv_scan/data`:

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

1. `split_audio_clips.py` splits one long audio file into per run/trial files in `stimuli/audio`
1. `copy_transcripts.py` copies and normalizes raw transcripts into `stimuli/transcript`
1. `copy_timings.py` copies and normalizes raw transcripts into `stimuli/timing`
1. `fix_transcripts.sh` fix transcription problems

1. `wordalign.py` uses whisperx wav2vec implementation to force-align

1. `process_transcripts.py` pipeline to normalize, sentencize, and tokenize transcripts and prepare for alignment
1. `align_transcript.sh` run forced-alignment to get word-level onset/offsets
1. `merge_transcripts.py` merge our utterance-level transcripts with forced aligner word-level

1. `qa.ipynb` for QA along the way

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

## dependencies

See `requirements.txt` and:

```
pip install accelerate himalaya nilearn scipy scikit-learn spacy tqdm transformers voxelwise_tutorials gensim pandas matplotlib seaborn torch torchaudio torchvision surfplot neuromaps git+https://github.com/m-bain/whisperx.git jupyter tqdm nltk
```

surfplot errors on jupyter
https://github.com/MICA-MNI/BrainSpace/issues/66

## brain 

ROIs
https://web.mit.edu/evlab/funcloc/
https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations

or use neuromaps to get these? but if i have to resample it anyways? can i resample with nilearn
or should i use 3dsample ?
```
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.nii
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.txt
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.nii
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.txt
wget https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_1000Parcels_Kong2022_17Networks_order_FSLMNI152_1mm.nii.gz
 ```