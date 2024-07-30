#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --job-name=scipy
#SBATCH --mem=90G
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-type=begin
#SBATCH -o dasktest.o.%J
#SBATCH -e dasktest.o.%J

# setup python env
# change to module/conda envs you have
module load anaconda3/2020.02
conda activate daxos

# cd to dir - assuming installed in home dir here
scriptdir="$HOME/daxg/scripts"
data_dir="/your/data/dir"

# convert raw to hdf5
# note covar file is in standard tab-delimited covar format for plink
python3 ${scriptdir}/adjust_hdf5_for_covars.py \
    --train ${data_dir}/train_file.hdf5 \
    --test ${data_dir}/test_file.hdf5 \
    --covar ${data_dir}/covar_file.txt \
    --out_train ${data_dir}/train_file_adjusted.hdf5 \
    --out_test ${data_dir}/test_file_adjusted.hdf5

