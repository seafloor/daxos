import pandas as pd
import numpy as np
import argparse
import os


def read_covars(f, verbose=True, sep='\t'):
    print('\n--> Reading covar file assuming plink format')
    cov = pd.read_csv(f, sep=sep)

    if verbose:
        print('Head of covar file:\n')
        print(cov.head())

    return cov


def check_covars(data):
    assert data.columns[0] == 'FID', 'Covariate file missing labelled FID column'
    assert data.columns[1] == 'IID', 'Covariate file missing labelled IID column'
    assert data.shape[1] > 2, 'Covariate file requires > 2 columns (minimum: FID, IID, COV1)'
    assert data.shape[0] > 1, f'Covariate file only has {data.shape[1]} rows'
    assert data.shape[0] == data['IID'].unique().shape[0], 'Not all IIDs unique in covariate file'
    assert data.shape[0] == data['FID'].unique().shape[0], 'Not all FIDs unique in covariate file'

    print(f'--> Read covar file of shape {data.shape}')
    print(f'Column names: {data.columns.to_numpy()}')
    print('Sample head:\n')
    print(data.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split IDs from covariate file',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('-f', '--file', type=str, default='',
                        help='Full path for input covariate file.')
    parser.add_argument('-p', '--proportion', type=float, default=0.7,
                        help='Train split fraction. Default 0.7.')
    parser.add_argument('-o', '--out_dir', type=str, default='',
                        help='Full path for ouput directory.')
    parser.add_argument('-s', '--seed', type=int, default=123,
                        help='Random seed for splits.')
    args = parser.parse_args()

    np.random.seed(args.seed)

    df = read_covars(args.file, verbose=False)
    check_covars(df)

    train = df.sample(frac=args.proportion)
    test = df.loc[~df['IID'].isin(train['IID'].to_numpy()), :]

    assert all(train['IID'].isin(df['IID'])), 'Splitting error: not all train IDs in original covariate file'
    assert all(test['IID'].isin(df['IID'])), 'Splitting error: not all test IDs in original covariate file'
    assert all(~train['IID'].isin(test['IID'])), 'Splitting error: some train/test IDs overlap'

    train['IID'].to_csv(os.path.join(args.out_dir, 'train', 'train_ids.txt'), index=False, header=False)
    test['IID'].to_csv(os.path.join(args.out_dir, 'test', 'test_ids.txt'), index=False, header=False)

    train.loc[:, ['FID', 'IID']].to_csv(os.path.join(args.out_dir, 'train', 'train_ids_plinkformat.txt'), index=False, header=False, sep='\t')
    test.loc[:, ['FID', 'IID']].to_csv(os.path.join(args.out_dir, 'test', 'test_ids_plinkformat.txt'), index=False, header=False, sep='\t')
