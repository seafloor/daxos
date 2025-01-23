#!/bin/bash                                                                                                               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=daskml
#SBATCH --mem=40G
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --mail-type=end 
#SBATCH --mail-type=fail
#SBATCH --mail-type=begin
#SBATCH -o dasktest.o.%J
#SBATCH -e dasktest.o.%J

# call with additional slurm flags in the sbatch submission
# e.g. sbatch --mail-user=email --account acctnum -p queuename submit_plink_conversion.sh

# set params for conversion
# assuming files are all in the  home dir
scriptdir="$HOME/daxos/scripts"
# sets the chunk size for reading plink files - recommended to avoid memory
# blowout when processed by numpy
nrow_read_raw=5000
# essential that this is reasonable. If N SNPs is in the hundreds of thousands then this or 
# even half it are good. But if you're running on a small regions then it can go up an order
# of magnitude. To be optimal, it should match or divide into the row chunk size used for downstream
# analysis (i.e. if you're going to use 150 later when reading file, using 100 now would force
# chunk reshuffling that impacts performance)
dask_chunk_size=100
# name of the plink file to convert
# should be plinkfile.bed/bim/fam
plinkfile='/your/directory/plinkfile'
# file showing the allele coding to use
# needed in case SNPs get flipped when you export separately from train and test splits
recodedallelefile='/your/allele/file'
dtype='float16'  # usually sufficient for snps, just can be changed to float32 if needed
plinkdir='/your/plink/location'  # directory where the plink executable is on your system

## handle tmp dir and job trap
trap "rm -r /tmp/user_${SLURM_JOB_ID}" SIGUSR1
mkdir /tmp/user_${SLURM_JOB_ID}
chmod 700 /tmp/user_${SLURM_JOB_ID}

# setup python env
# change to whatever module/conda environments you have
module load conda/23.11-py311
conda activate daxos

# convert to raw with plink
if [ -s "${plinkfile}.raw" ]
then
    echo "--> File ${plinkfile}.raw already exists - skipping plink conversion"
else
    ${plinkdir}/plink --bfile $plinkfile --recode A --recode-allele $recodedallelefile --out $plinkfile
fi

# shuffle on command line
if [ -s "${plinkfile}_shuffled.raw" ]
then
    echo "--> File ${plinkfile}_shuffled.raw already exists - skipping row shuffling."
else
    awk '(NR == 1) {print $0}' ${plinkfile}.raw > ${plinkfile}_shuffled.raw
    awk '(NR > 1) {print $0}' ${plinkfile}.raw | shuf >> ${plinkfile}_shuffled.raw
fi

# set nrows from fam file
wc_fam=($(wc -l ${plinkfile}.fam))
nrow_fam=${wc_fam[0]}

# convert raw to hdf5
python3 ${scriptdir}/convert.py \
    --in_raw ${plinkfile}_shuffled.raw \
    --out_hdf5 ${plinkfile}_shuffled \
    --nrows ${nrow_fam} \
    --dask_chunks ${dask_chunk_size} \
    --read_raw_in_chunks True \
    --read_raw_chunk_size ${nrow_read_raw} \
    --dtype ${dtype}

# remove dask files
rm -r /tmp/user_${SLURM_JOB_ID}
