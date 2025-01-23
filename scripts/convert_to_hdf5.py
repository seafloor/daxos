import sys
import os
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from daxos import read
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert plink files to dask hdf5 files for machine learning',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--in_raw', type=str, default='',
                        help='Full path for input raw file')
    parser.add_argument('--out_hdf5', type=str, default='',
                        help='Full path for output hdf5 file')
    parser.add_argument('--nrows', type=int,
                        help='Number of individuals (nrow of fam file)')
    parser.add_argument('--dask_chunks', type=int, default=100,
                        help='Size of chunks in hdf5 file')
    parser.add_argument('--read_raw_chunk_size', type=int, default=1000,
                        help='Number of rows to read from raw file in one go.')
    parser.add_argument('--dtype', type=str, default='float16',
                        help='Numpy dtype to use for all columns. Must be in ["float16", "float32", "float64"].')

    args = parser.parse_args()

    if args.dtype not in ['float16', 'float32', 'float64']:
        raise ValueError('--dtype not recognised')

    print('\n--> Starting raw conversion to hdf5 with params: {}'.format(args))

    read.raw_to_hdf5(args.in_raw, args.out_hdf5, args.nrows, row_chunks=args.dask_chunks, check_output=True,
                     read_raw_chunk_size=args.read_raw_chunk_size, dtype=args.dtype)
