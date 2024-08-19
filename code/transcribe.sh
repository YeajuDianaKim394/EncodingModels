#!/usr/bin/env bash
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=4G                 # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=transcribe    # create a short name for your job
#SBATCH --gres=gpu:1             # get a gpu
#SBATCH -o 'logs/%A-transcribe.log'
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zzada@princeton.edu

# took 30 min to transcribe all stranger's generate conditions

# to download the model offline
# git clone https://huggingface.co/guillaumekln/faster-whisper-large-v2 large-ct2
# git lfs pull --include=model.bin

source /usr/share/Modules/init/bash
module load anaconda3/2023.3
conda activate fconv

echo "${CONDA_PROMPT_MODIFIER}Transcribe"
echo "${CONDA_PROMPT_MODIFIER}Requester: $USER"
echo "${CONDA_PROMPT_MODIFIER}Node: $HOSTNAME"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "${CONDA_PROMPT_MODIFIER}Start time:" `date`

subjects=(104 105 106 107 108 111 112 114 116 117 120 122 123 126 128 129 131 132 133 137 138 142 143 153 156 157 158 163 174)

whisperx \
    --model models/large-ct2 \
    --output_dir data/stimuli/whisperx \
    --output_format json \
    --task transcribe \
    --language en \
    --diarize \
    --min_speakers 2 --max_speakers 2 \
    --hf_token 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 \
    --device cuda \
    data/stimuli/conv-*/audio/*condition-G*wav

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`