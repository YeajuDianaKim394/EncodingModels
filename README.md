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

order:
1. `split_audio_clips.py`
  1. `transcribe.sh`
    1. `move_whisper_transcipts.py`
      1. `embeddings.py`
      2. `feature_gen.py` articulatory
      3. `feature_gen.py` syntactic
  2. `feature_gen.py` spectral
2. `clean.py` -m trialmot9 and runmot24 [INFLUENCED]
3. encoding? [INFLUENCED]

### transcript preprocessing
We have transcripts and audio at the trial level. Each transcript contains utterances per speaker turn and the utterance onset.

1. `split_audio_clips.py` splits one long audio file into per run/trial files in `stimuli/audio`
2. `transcribe.sh`
3. `move_whisper_transcipts.py`

if using rev transcripts:

2. `copy_timings.py` copies and normalizes raw transcripts into `stimuli/timing`
3. `copy_transcripts.py` copies and normalizes raw transcripts into `stimuli/transcript`
4. `fix_transcripts.sh` fix transcription problems
5. `wordalign.py` uses whisperx wav2vec implementation to force-align
6. `qa.ipynb` for QA along the way

`rsync -av scotty.princeton.edu:/jukebox/hasson/zaid/fmri-convs/data/stimuli data/`

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
pip install accelerate himalaya nilearn scipy scikit-learn spacy tqdm transformers voxelwise_tutorials gensim pandas matplotlib seaborn torch torchaudio torchvision surfplot neuromaps git+https://github.com/m-bain/whisperx.git jupyter tqdm nltk statsmodels h5py netneurotools pyrcca openpyxl ai2-olmo
```

wget http://lxcenter.di.fc.ul.pt/wn2vec.zip

surfplot errors on jupyter
https://github.com/MICA-MNI/BrainSpace/issues/66

## brain 

ROIs
https://web.mit.edu/evlab/funcloc/
https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations
glasser

```
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.nii
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.txt
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.nii
wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.txt
wget https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_1000Parcels_Kong2022_17Networks_order_FSLMNI152_1mm.nii.gz
 ```

 ~surf.py:300 to be able to plot a 2D map~
 ```
298                     if hasattr(cm, '_lut'):
  1                         table = cm._lut * 255
  2                     else:
  3                         # nvals = lut['numberOfTableValues']
  4                         nvals = cm.N
  5                         table = cm(np.linspace(0, 1, nvals)) * 255
  6                     table = table.astype(np.uint8)
 ```
 ~surfplot/plotting.py:461
 ```
    def _add_colorbars(self, location='bottom', label_direction=None,   
                       n_ticks=3, decimals=2, fontsize=10, draw_border=True, 
                       outer_labels_only=False, aspect=20, pad=.08, shrink=.3, 
                       fraction=.05, fig=None, ax=None):
            fig = plt if fig is None else fig
            ax = ax if ax is not None else plt.gca()
            cb = fig.colorbar(sm, ticks=ticks, location=location, 
                              fraction=fraction, pad=cbar_pads[i], 
                              shrink=shrink, aspect=aspect, ax=ax)


 ```