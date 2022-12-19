#!/bin/bash
# usage: sbatch scripts/sbatch/run_tune_expt.sbatch EXPERIMENT=mooc_course
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=100000m
#SBATCH --time=05-00:00:00
#SBATCH --account=efml
#SBATCH --chdir=/gscratch/efml/jpgard/tableshift
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jpgard@cs.washington.edu
#SBATCH --job_name=run_tune_expt_${EXPERIMENT}

date;hostname;id;pwd

echo 'experiment is:' $EXPERIMENT

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tableshift
which python

## experimental command to test on a single experiment iteration
#python scripts/ray_train_cached.py \
#  --model_name mlp \
#  --uid acsincomedomain_split_varname_DIVISIONdomain_split_ood_value_00

## the actual command to run
python experiments/domain_shift.py \
  --experiment $EXPERIMENT \
  --use_cached \
  --time_budget_hrs 1