#!/usr/bin/env bash
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=8G                 # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=enc           # create a short name for your job
#SBATCH --array=001,004,005,006,007,012,014,020,023,026,057,101,104,105,106,107,112,114,120,123,126,157
#SBATCH -o 'encoding/logs/%A_%a.log'      # write to a log file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzada@princeton.edu

# 001, 004, 005, 006, 007, 012, 014, 020, 023, 026, 057, 101, 104, 105, 106, 107, 112, 114, 120, 123, 126, 157

source /usr/share/Modules/init/bash
module load anacondapy/2022.05
conda activate fb2b

echo "${CONDA_PROMPT_MODIFIER}Embeddings"
echo "${CONDA_PROMPT_MODIFIER}Requester: $USER"
echo "${CONDA_PROMPT_MODIFIER}Node: $HOSTNAME"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "${CONDA_PROMPT_MODIFIER}Start time:" `date`

python code/encoding.py -s "$SLURM_ARRAY_TASK_ID" -j 3

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`