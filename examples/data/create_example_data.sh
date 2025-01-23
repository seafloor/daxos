#! /bin/bash
# example script showing how dummy test data were created sith setup_data.py file
# change module load for runs on different systems
module load conda/23.11-py311 
conda activate daxos

python3 setup_data.py --seed 1 --n 10000 --output_file example_data_train.hdf5
python3 setup_data.py --seed 2 --n 5000 --output_file example_data_test.hdf5