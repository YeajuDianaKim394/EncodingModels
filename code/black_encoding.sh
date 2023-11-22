#!/usr/bin/env bash
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=4G                 # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=blenc         # create a short name for your job
#SBATCH --gres=gpu:1             # get a gpu
#SBATCH -o 'logs/%A_black.log'
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zzada@princeton.edu

source /usr/share/Modules/init/bash
module load anaconda3/2023.3 cudatoolkit/11.7 cudnn/cuda-11.x/8.2.0
conda activate fconv

echo "${CONDA_PROMPT_MODIFIER}Encoding"
echo "${CONDA_PROMPT_MODIFIER}Requester: $USER"
echo "${CONDA_PROMPT_MODIFIER}Node: $HOSTNAME"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "${CONDA_PROMPT_MODIFIER}Start time:" `date`

# TQDM_DISABLE=1

python code/black_encoding.py

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`