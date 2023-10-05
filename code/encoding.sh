#!/usr/bin/env bash
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=5G                 # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=enc           # create a short name for your job
#SBATCH --gres=gpu:1             # get a gpu
#SBATCH --array=0                # 0 does all subjects, 1 does 0XX and 2 does 1XX
#SBATCH -o 'encoding/logs/%A_%a.log'
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

# bad, and good timings [now fixed tho]
# bads=(4 22 29 38 43 104 122 129 138 143)
# strangers=(05 06 07 08 11 12 14 16 17 20 23 26 28 31 32 33 37 42 53 56 57 58 63 74 105 106 107 108 111 112 114 116 117 120 123 126 128 131 132 133 137 142 153 156 157 158 163 174)

strangers=(04 05 06 07 08 11 12 14 16 17 20 22 23 26 28 29 31 32 33 37 38 42 43 53 56 57 58 63 74 104 105 106 107 108 111 112 114 116 117 120 122 123 126 128 129 131 132 133 137 138 142 143 153 156 157 158 163 174)

subjects=("${strangers[@]}")
if [ -n "$TEST" ]; then
    if [ "$TEST" -eq 1 ]; then
        subjects=("${strangers[@]:0:29}")
    elif [ "$TEST" -eq 2 ]; then
        subjects=("${strangers[@]:29}")
    fi
fi

for sub in "${subjects[@]}"; do
    echo $sub
    python code/encoding.py -s "$sub" -j 1 -m model-gpt2-medium_layer-0.75
done

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`