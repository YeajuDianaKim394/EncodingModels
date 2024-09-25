#!/usr/bin/env bash
#SBATCH --partition=mig
#SBATCH --time=00:62:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=4G                 # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=blenc         # create a short name for your job
#SBATCH --gres=gpu:1             # get a gpu
#SBATCH --array=0,21             # layer
#SBATCH -o 'logs/%A_%a_black.log'
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zzada@princeton.edu

source /usr/share/Modules/init/bash
module load anaconda3/2023.3
conda activate fconv

echo "${CONDA_PROMPT_MODIFIER}Encoding"
echo "${CONDA_PROMPT_MODIFIER}Requester: $USER"
echo "${CONDA_PROMPT_MODIFIER}Node: $HOSTNAME"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "${CONDA_PROMPT_MODIFIER}Start time:" `date`

export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

modelname=olmo-7b
modelname=contextual
layer="$SLURM_ARRAY_TASK_ID"

python code/black_encoding.py -m "$modelname" --layer="$layer"

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`