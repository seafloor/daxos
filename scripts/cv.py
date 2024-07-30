import sys
import os
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxg'))
from daxg.utils import parse_bool
from daxg.distribute import spin_cluster, scale_cluster
from daxg.crossvalidate import cv_xgb
from daxg.read import read_ml
from dask.distributed import Client
import pathlib
import argparse
import numpy as np
import time
import h5py


def main(client, X, y, iter_search=30, boost_rounds=1000, folds=5, cv_subsample=0, out_dir=None, out_prefix=None,
         score_method='AUC', min_subsample=0.7, max_subsample=0.7, **fit_kwargs):
    assert all([x is not None for x in (out_dir, out_prefix)])

    best_params, best_score, cv_scores, y_pred, param_grid = cv_xgb(
        client, X, y, cv_subsample, folds, iter_search, boost_rounds, min_subsample,
        max_subsample, score_method, **fit_kwargs
    )

    print(f"Best params {best_params}")
    print(f"{score_method} from best cross-validation round (still inflated, but more accurate): {best_score:.2g}")

    return cv_scores, y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Dask and XGBoost',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--in_ml', type=str, default='',
                        help='.hdf5 file with "/x" and "/y" as da.arrays, and "/rows" and "/cols" as pd.DataFrames.')
    parser.add_argument('--n_iter_search', type=int, default=30,
                        help='Number of iterations of cross-validated random search.')
    parser.add_argument('--n_boost_round', type=int, default=100,
                        help='Number of boosting rounds in XGBoost. Learning rate search/values should be changed too.'
                             ' Note that this is the total number of boosting rounds, not just the number in this fit.')
    parser.add_argument('--n_threads_per_worker', type=int, default=5,
                        help='Number of threads used - passed to Dask and XGBoost for multi-threading.')
    parser.add_argument('--time_per_worker', type=str, default='05:00:00',
                        help='Time for compute nodes in dsitributed cluster - should be shorter than scheduler time.')
    parser.add_argument('--mem_per_worker', type=str, default='30GB',
                        help='Total amount of RAM per node (not per CPU). Will be same for all n_workers_in_cluster.')
    parser.add_argument('--n_workers_in_cluster', type=int, default=4,
                        help='Number of compute nodes in SlurmCluster. Passed to cluster.scale(n_workers_in_cluster).')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds in cross-validation.')
    parser.add_argument('--out', type=str,
                        help='Full path for directory to save output files.')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for output filename.')
    parser.add_argument('--local_dir', type=str, default='',
                        help='Local directory for dask workers.')
    parser.add_argument('--cv_subsample_size', type=int, default=0,
                        help='Number of observations to be used in CV. Uses all data if 0 is given.')
    parser.add_argument('--row_chunk_size', type=int, default=100,
                        help='Size of the chunks when reading in dask arrays.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed passed to numpy. Not set if 0 is given.')
    parser.add_argument('--cluster', type=str, default='local',
                        help='Run local cluster on current node or use as the root to spin-up a distributed cluster.')
    parser.add_argument('--xgb_eval_metric', type=str, default='logloss',
                        help='Used to evaluate XGB boosting rounds. Can also use aucpr.')
    parser.add_argument('--min_subsample', type=float, default=0.7,
                        help='Search subsample between min_subsample and max_subsample in HP tuning.')
    parser.add_argument('--max_subsample', type=float, default=0.7,
                        help='Search subsample between min_subsample and max_subsample in HP tuning.')
    parser.add_argument('--verbose', type=str, default='False',
                        help='If true include print of loss per boosting round.')
    parser.add_argument('--loss', type=str, default='binary:logistic',
                        help='Will be overriden with "reg:squarederror" if covar supplied')
    parser.add_argument('--gpu', type=str, default='False',
                        help='Use GPUs if True. Not implemented yet.')
    parser.add_argument('--interface', type=str, default='ib0',
                        help='Networking interface for connection between workers. Uses "lo" if --cluster is "local"')
    parser.add_argument('--xkey', type=str, default='x',
                        help='Key in hdf5 file for X.')
    parser.add_argument('--ykey', type=str, default='y',
                        help='Key in hdf5 file for y.')
    parser.add_argument('--worker_queue', type=str, default='None',
                        help='Name of queue to submit worker jobs to. Default None.')
    args = parser.parse_args()

    verbose, gpu = [parse_bool(x) for x in (args.verbose, args.gpu)]
    interface, queue = [None if x.lower() == 'none' else x for x in (args.interface, args.worker_queue)]

    if args.ykey == 'y_adjusted':
        loss = 'reg:squarederror'
        platt_scale = True
        eval_metric = 'rmse'
        score_method = 'RMSE'
    else:
        loss = args.loss
        platt_scale = False
        eval_metric = args.xgb_eval_metric
        score_method = 'AUC'

    if gpu:
        raise NotImplementedError('GPUs not supported yet')

    t0 = time.time()
    if args.seed > 0:
        np.random.seed(args.seed)

    with spin_cluster(args.cluster, args.n_threads_per_worker, args.local_dir, 1, args.mem_per_worker,
                      args.time_per_worker, interface, queue) as cluster:
        scale_cluster(cluster, args.cluster, args.n_workers_in_cluster, args.n_threads_per_worker, args.mem_per_worker)

        with Client(cluster) as client:
            print(f'Returned client with info/address: {client}')
            with h5py.File(args.in_ml) as f:
                X, y, rows, columns = read_ml(args.in_ml, f, row_chunks=args.row_chunk_size,
                                              x_key=args.xkey, y_key=args.ykey)

                fit_kwargs = dict(zip(['n_threads', 'eval_metric', 'loss', 'verbose'],
                                      [args.n_threads_per_worker, eval_metric, loss, verbose]))

                client.wait_for_workers(n_workers=args.n_workers_in_cluster)

                scores, y_pred = main(
                    client, X, y, args.n_iter_search, args.n_boost_round, args.n_folds, args.cv_subsample_size,
                    args.out, args.prefix, score_method, args.min_subsample, args.max_subsample, **fit_kwargs
                )

            cv_out = os.path.join(args.out, 'cv', 'hp_search', f'{args.prefix}_cv_scores.csv')
            y_pred_out = os.path.join(args.out, 'cv', 'y_pred', f'{args.prefix}_best_cv_ypred.csv')
            _ = [pathlib.Path(x).parent.mkdir(parents=True, exist_ok=True) for x in (cv_out, y_pred_out)]
            scores.to_csv(cv_out, index=False)
            y_pred.assign(IID=rows.iloc[y_pred['y_idx'].to_numpy(), 1].to_numpy()).to_csv(y_pred_out, index=False)

    t1 = time.time()
    t2 = t1 - t0
    print(f'\nTime taken to process on CPU: '
          f'{t2 // 3600 % 24:.2f} hours, {t2 // 60 % 60:.2f} minutes, {t2 % 60:.2f} seconds')
