#!/usr/bin/env bash
#SBATCH --time=01:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=mfa           # create a short name for your job
#SBATCH --array=104          # job array with index values 0, 1, 2, 3, 4
#SBATCH -o 'staging/logs/%A_%a.log'      # write to a log file

# Force-align transcripts to audo
# https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-align-pretrained


strangers="101 104 105 106 107 108 111 112 114 116 117 119 120 122 123 126 128 129 131 132 133 137 138 142 143 153 156 157 158 163 171 174"
friends="103 109 113 118 121 127 130 134 145 146 147 148 150 151 152 154 155 159 160 161 162 164 165 166 167 169 170 172 173 175"

# removed bad convs 119, 168, 171
strangers="101 104 105 106 107 108 111 112 114 116 117 120 122 123 126 128 129 131 132 133 137 138 142 143 153 156 157 158 163 174"

# source /usr/share/Modules/init/bash
# module load anacondapy/2022.05
# conda activate fb2b

export MFA_ROOT_DIR="/scratch/zzada/MFA"

# Choose which acoustic/dictionary model to use
model="english_mfa"

# # First time, download the models
# mfa model download acoustic $model
# mfa model download dictionary $model

mfa configure --disable_auto_server
# mfa server start

# Run alignment on one conversation
# Options:
#   --overwrite             Overwrite output files when they exist
#   --clean                 Remove files from previous runs
#   --include_original_text Flag to include original utterance text in the output.
#   --use_mp, --no_use_mp   Turn on/off multiprocessing. Multiprocessing is recommended will allow for faster executions.
#   --debug, --no_debug     Turn on/off debugging checks
#   --single_speaker        Single speaker mode creates multiprocessing splits based on utterances rather than speakers
#   --num_jobs 2
#   --beam 100 --retry_beam 400

stagedir="staging/audio"
tempdir="staging/mfa/"

mkdir -p "$stagedir"
mkdir -p "$tempdir"

# set -e

if [[ $# -gt 0 ]]; then
    echo "Using conversations from arguments"
    convs="$@"
elif [[ -v $SLURM_ARRAY_TASK_ID ]]; then
    echo "Using conversations from slurm"
    convs="$SLURM_ARRAY_TASK_ID"
else
    # echo "Using all conversations in stimuli/"
    # convs="$(find stimuli -maxdepth 1 -type d -name 'conv-*' -exec sh -c 'basename {} | cut -d- -f2' \;)"
    echo "Using strangers"
    convs=$strangers
fi

for convid in $convs; do
    echo "on conv-$convid $(date -Iminutes)"
    for file in $(ls -1 stimuli/conv-$convid/transcript/*.TextGrid); do
        echo $file
        base="$(basename $file .TextGrid | head -c 50)"
        datadir="$stagedir/$base"
        mkdir -p $datadir
        ln -fs $PWD/$file $datadir
        audiofile=${file/transcript/audio}
        ln -fs $PWD/${audiofile%.*}.wav $datadir
        outdir="stimuli/conv-$convid/aligned"
        # mfa validate --temporary_directory $temp_dir $data_dir $model $model
        echo mfa align --overwrite --single_speaker --temporary_directory $tempdir $datadir $model $model $outdir --beam 1000 --retry_beam 4000
    done
done

# mfa server stop
