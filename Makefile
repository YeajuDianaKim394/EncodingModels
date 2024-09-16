
make-env:
	pip install accelerate himalaya nilearn scipy scikit-learn spacy tqdm transformers voxelwise_tutorials gensim pandas matplotlib seaborn torch torchaudio torchvision surfplot neuromaps git+https://github.com/m-bain/whisperx.git jupyter tqdm nltk statsmodels h5py netneurotools openpyxl natsort

download-atlas:
	# may not be needed. but need to run atlas.ipynb.
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.nii
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_MD_HE197.txt
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.nii
	wget https://web.mit.edu/evlab//assets/funcloc_assets/allParcels_language_SN220.txt
	wget https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_1000Parcels_Kong2022_17Networks_order_FSLMNI152_1mm.nii.gz

split-audio:
	python code/split_audio_clips.py

whisper_model:
	git clone https://huggingface.co/guillaumekln/faster-whisper-large-v2 large-ct2
	git lfs pull --include=model.bin

transcribe:
	# depends on whisper_model and set HF_TOKEN env variable
	sbatch --job-name=transcribe --mem=4G --time=00:30:00 --gres=gpu:1 code/slurm.sh -- \
		whisperx \
			--model models/large-ct2 \
			--output_dir data/stimuli/whisperx \
			--output_format json \
			--task transcribe \
			--language en \
			--diarize \
			--min_speakers 2 --max_speakers 2 \
			--hf_token "${HF_TOKEN}" \
			--device cuda \
			data/stimuli/conv-*/audio/*condition-G*wav

process_trancsripts:
	python code/move_whisper_transcripts.py

llm_embeddings:
	sbatch --job-name=emb --mem=8G --time=00:05:00 --gres=gpu:1 code/slurm.sh -- \
		code/embeddings.py -m gpt2-2b --layer 24
	# python code/embeddings.py -m gpt2-2b --layer 0

generate_features:
	python code/feature_gen.py spectral
	python code/feature_gen.py articulatory
	python code/feature_gen.py syntactic

confound_regression:
	python code/clean.py -m default_task

# encoding:
