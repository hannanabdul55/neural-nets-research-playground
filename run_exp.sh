#!/bin/bash
#
#SBATCH --job-name=exp_norm_lr
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res%j.err        # File to which STDERR will be written
#SBATCH --partition=2080ti-long   # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-12:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=4gb    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu
export PYTHONPATH=/home/akanji/neural-nets-research-playground:$PYTHONPATH
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
#conda activate norm_exp
/home/akanji/miniconda3/envs/norm_exp/bin/python  layernorm_experiment.py "$@"

sleep 1
