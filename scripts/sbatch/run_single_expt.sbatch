#!/bin/bash
# usage:
# sbatch --export=EXPERIMENT='acsfoodstamps_region',MODEL='resnet',\
# UID='acsincomedomain_split_varname_DIVISIONdomain_split_ood_value_00' \
# scripts/sbatch/run_tune_expt.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=100000m
#SBATCH --time=07-00:00:00
#SBATCH --account=efml
#SBATCH --chdir=/gscratch/efml/jpgard/tableshift
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jpgard@cs.washington.edu

date;hostname;id;pwd

echo 'experiment is:' $EXPERIMENT

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tableshift
which python

## experimental command to test on a single experiment iteration
python scripts/ray_train.py \
  --model_name $MODEL \
  --uid $UID \
  --num_samples 100 \
  --use_cached
