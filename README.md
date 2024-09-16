# brain-to-brain hyperscanned fMRI conversations

## code

order:
1. `split_audio_clips.py`
  1. `transcribe.sh`
    1. `move_whisper_transcipts.py`
      1. `embeddings.py`
      2. `feature_gen.py` articulatory
      3. `feature_gen.py` syntactic
  2. `feature_gen.py` spectral
2. `clean.py` -m trialmot9 and runmot24
3. encoding?

other:
* `constants.py`


## dependencies
See `requirements.txt` and Makefile `make-env` target.

* surfplot errors on jupyter: https://github.com/MICA-MNI/BrainSpace/issues/66

modified surfplot `~surfplot/plotting.py:461`:
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