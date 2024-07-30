#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --job-name=dask-boruta
#SBATCH --mem=50GB
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-type=begin
#SBATCH --output=job.%j.log
#SBATCH --error=job.%j.err

# set to wherever installed
scriptdir="$HOME/daxg/scripts"

# start timer and log run info
start=`date +%s`

# setup python env
# set to module/conda envs you have
module load anaconda3/2020.02
conda activate daxos

# run boruta on shap values
python3 ${scriptdir}/boruta_shap.py

# finish timer
end=`date +%s`
runtime=$((end-start))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$(( (runtime % 3600) % 60 ))
echo "Total runtime: $hours:$minutes:$seconds (hh:mm:ss)"
echo -e "\n############################## RUN COMPLETE :) #################################\n"
