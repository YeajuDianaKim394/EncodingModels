#!/usr/bin/env bash
#SBATCH --time=00:25:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=8G                 # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=enc           # create a short name for your job
#SBATCH --array=05,07,08,11,12,14,16,17,20,22,23,26,28,29,31,32,33,37,38,42,43,53,56,57,58,63,74,105,106,107,111,112,114,116,117,120,122,123,126,128,129,131,132,133,137,138,142,143,153,156,157,158,163,174,
#SBATCH -o 'encoding/logs/%A_%a.log'      # write to a log file
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zzada@princeton.edu

# strangers
# 04,05,06,07,08,11,12,14,16,17,20,22,23,26,28,29,31,32,33,37,38,42,43,53,56,57,58,63,74,104,105,106,107,108,111,112,114,116,117,120,122,123,126,128,129,131,132,133,137,138,142,143,153,156,157,158,163,174,
# 104,105,106,107,108,111,112,114,116,117,120,122,123,126,128,129,131,132,133,137,138,142,143,153,156,157,158,163,174,

source /usr/share/Modules/init/bash
module load anacondapy/2022.05
conda activate fb2b

echo "${CONDA_PROMPT_MODIFIER}Embeddings"
echo "${CONDA_PROMPT_MODIFIER}Requester: $USER"
echo "${CONDA_PROMPT_MODIFIER}Node: $HOSTNAME"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "${CONDA_PROMPT_MODIFIER}Start time:" `date`

python code/encoding.py -s "$SLURM_ARRAY_TASK_ID" -j 3 -m gpt2

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`