import argparse
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from daxos.read import subset_hdf5s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subset hdf5 files by rows/columns',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--in_path', type=str,
                        help='Full path for existing hdf5 file')
    parser.add_argument('--out_path', type=str,
                        help='Full path for hdf5 file to write to')
    parser.add_argument('--snps', default=None, type=str,
                        help='Path to text file of SNP names (one row per SNP)')
    parser.add_argument('--ids', default=None, type=str,
                        help='Path to IIDs (one row per ID)')
    parser.add_argument('--xkey', default='x', type=str,
                        help='The key for genotype data in the hdf5 file')
    parser.add_argument('--ykey', default='y', type=str,
                        help='The key for the outcome in the hdf5 file')
    args = parser.parse_args()

    snps = pd.read_csv(args.snps, header=None, squeeze=True).to_numpy() if args.snps is not None else None
    ids = pd.read_csv(args.ids, header=None, squeeze=True, dtype=str).to_numpy() if args.ids is not None else None

    subset_hdf5s(in_path=args.in_path, out_path=args.out_path, x_key=args.xkey, y_key=args.ykey,
                 row_ids=ids, col_ids=snps)
