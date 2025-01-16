import sys
import os
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from daxg.read import read_ml, save_booster
from daxg.explain import collect_importances, subset_predictors
from daxg.crossvalidate import read_hp_search_results, incremental_fit_xgb, fit_dask_xgb, persist_daskdmatrix, score_model
from daxg.utils import parse_bool
from daxg.scale import run_cv_and_platt_scale
import xgboost as xgb
from dask.distributed import Client
from daxg.distribute import spin_cluster, scale_cluster
import dask.array as da
import pathlib
import joblib
import argparse
import pandas as pd
import numpy as np
import pprint
import time
import h5py


def main(client, X, y, colnames, n_folds=5, increment_refit=False, row_chunks=100, out_dir=None, out_prefix=None,
         incremental_start_round=1, incremental_n_boost_per_round=1, run_shap_main=False, run_shap_inter=False,
         hp_search_file='', platt_scale=False, y_binary=None, score_method='AUC',
         n_booster_overide=None, **fit_kwargs):

    # read CV results and assume sort order is descending only if using AUC, else ascending (e.g. for RMSE)
    best_params = read_hp_search_results(hp_search_file)
    if n_booster_overide is not None:
        print(f"Manually overwriting n_boosters from {best_params['n_boost_round']} to {n_booster_overide}")
        best_params['n_boost_round'] = n_booster_overide

    print("\n--> Fitting/Refitting on full dataset with params:")
    pprint.pprint(best_params)

    # refit using best params from hp_search_file, either single fit in-memory, or incrementally to reduce memory use
    if increment_refit:
        # drop arg requirements for incremental refit
        # refactor names to be smaller
        # can pass these arguments as a dict because they are specific to incremental learning
        incremental_outdir = os.path.join(out_dir, 'refit')
        dtrain, bst, history, best_params = incremental_fit_xgb(
            client, X, y, colnames, best_params, incremental_start_round, incremental_outdir, out_prefix,
            incremental_n_boost_per_round, row_chunks, **fit_kwargs
        )
    else:
        print('\nCreating DMatrix for XGB')
        dtrain = persist_daskdmatrix(client, X, y, feature_names=colnames)

        print('\nTraining...')
        bst, history = fit_dask_xgb(client, dtrain, best_params, **fit_kwargs)

        print('\n--> Saving XGBoost model from first refit')
        first_refit_save_dir = os.path.join(out_dir, 'refit', 'models', f'{out_prefix}_origrefit_xgbmodel.json')
        save_booster(bst, first_refit_save_dir)

    print("\n--> Fitting/Refitting again on reduced predictor space")
    # perform a second refit. This first subsets the data to only the predictors used in the original refit,
    # then it fits a new model on the reduced data. Model will be more parsimonious and much easier to use with SHAP
    del dtrain  # removes the original X from the distributed compute nodes
    xgboost_refit_predictors = np.array(list(bst.get_score(importance_type='cover').keys()))
    X_refit, columns_used_in_refit = subset_predictors(X, colnames, xgboost_refit_predictors)

    # save column names for future predictions
    print('\nSaving columns used to refit XGBoost (only use to subset data when loading the saved booster)')
    cols_save_dir = os.path.join(out_dir, 'refit', 'predictors', f'{out_prefix}_used_cols.csv')
    pd.Series(columns_used_in_refit).to_frame().to_csv(cols_save_dir, index=False)
    if os.path.exists(cols_save_dir):
        print(f'Saved to path: {cols_save_dir}')

    print('\nCreating reduced DMatrix for XGB and SHAP')
    dtrain = persist_daskdmatrix(client, X_refit, y, feature_names=columns_used_in_refit)

    print('\nTraining on reduced DMatrix...')
    bst, history = fit_dask_xgb(client, dtrain, best_params, **fit_kwargs)

    print('\n--> Predicting back on the train set (internal validation, results are biased)...')
    prediction = xgb.dask.predict(client, bst, dtrain)
    train_set_score = score_model(y, prediction, score_method=score_method)

    print(f"{score_method} on train set (biased - cannot be used to measure performance): {train_set_score:.2g}")

    print('\n--> Saving XGBoost model from second refit')
    second_refit_save_dir = os.path.join(out_dir, 'refit', 'models', f'{out_prefix}_shaprefit_xgbmodel.json')
    save_booster(bst, second_refit_save_dir)

    # calculate built-in importance scores from XGBoost (gain etc.) as well as SHAP main and interaction effects
    # mean(|SHAP|) saved with importance scores in CSV; raw SHAP values saved to Zarr files because of multi-threading
    importance_outdir = os.path.join(out_dir, 'refit', 'importances')
    importance = collect_importances(client, bst, dtrain, columns_used_in_refit, importance_outdir, out_prefix,
                                     run_shap_main=run_shap_main, run_shap_inter=run_shap_inter)

    # if --covar file is passed and regressed off y, then predictions from XGB are no longer constrained to [0, 1]
    # if platt_scale then fit a model (logistic regression) and save the model for prediction on future data
    if platt_scale:
        print('\n--> Starting Platt Scaling of predictions')
        assert y_binary is not None, 'Supply original (unadjusted) y to y_binary if using platt_scale=True'
        del dtrain
        platt_model = run_cv_and_platt_scale(client, X_refit, y, y_binary, best_params, n_folds, score_method,
                                             manually_map_to_workers=True, **fit_kwargs)
        platt_save_dir = os.path.join(out_dir, 'refit', 'models', f'{out_prefix}_shaprefit_plattscalemodel.json')
        joblib.dump(platt_model, platt_save_dir)
        if os.path.exists(platt_save_dir):
            print(f'Model for platt-scaling in train set saved to: {platt_save_dir}')

    return bst, prediction, importance, columns_used_in_refit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Dask and XGBoost',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--in_ml', type=str, default='',
                        help='.hdf5 file with "/x" and "/y" as da.arrays, and "/rows" and "/cols" as pd.DataFrames.')
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
    parser.add_argument('--row_chunk_size', type=int, default=100,
                        help='Size of the chunks when reading in dask arrays.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed passed to numpy. Not set if 0 is given.')
    parser.add_argument('--cluster', type=str, default='local',
                        help='Run local cluster on current node or use as the root to spin-up a distributed cluster.')
    parser.add_argument('--incremental_learning', type=str, default='False',
                        help='If True, save the model after each boosting round and reload for the next round.'
                             ' DaskDMatrices for the refit are only created for each tree. This is only worthwhile'
                             ' doing if subsample is fixed to less than 1, meaning the full dataset is never loaded'
                             ' into memory. Use if full dataset is too big for RAM. Only performed for the refit.')
    parser.add_argument('--incremental_start_round', type=int, default=1,
                        help='Boosting round to start with. Used to continue previous incremental learning.')
    parser.add_argument('--incremental_n_boost_per_round', type=int, default=1,
                        help='Number of boosting rounds to run on each increment.')
    parser.add_argument('--xgb_eval_metric', type=str, default='logloss',
                        help='Used to evaluate XGB boosting rounds. Can also use aucpr.')
    parser.add_argument('--verbose', type=str, default='False',
                        help='If true include print of loss per boosting round.')
    parser.add_argument('--run_shap_main', type=str, default='False',
                        help='If true, generate SHAP values')
    parser.add_argument('--run_shap_inter', type=str, default='False',
                        help='If true, generate SHAP interaction values')
    parser.add_argument('--hp_search_results', type=str, default='',
                        help='Full path of CSV file containing results from cross-validation during hyperparameter '
                             'search. Requires columns ["AUC", "eta", "subsample", "colsample_bytree", "max_depth"]')
    parser.add_argument('--loss', type=str, default='binary:logistic',
                        help='Will be overriden with "reg:squarederror" if covar supplied')
    parser.add_argument('--gpu', type=str, default='False',
                        help='Use GPUs if True.')
    parser.add_argument('--interface', type=str, default='ib0',
                        help='Networking interface for connection between workers. Uses "lo" if --cluster is "local"')
    parser.add_argument('--xkey', type=str, default='x',
                        help='Key in hdf5 file for X.')
    parser.add_argument('--ykey', type=str, default='y',
                        help='Key in hdf5 file for y.')
    parser.add_argument('--worker_queue', type=str, default='None',
                        help='Name of queue to submit worker jobs to. Default None.')
    parser.add_argument('--n_booster_overide', type=int, default=None,
                        help='N booster for non-incremental learning. Overwrites HP-tuning value. Default None.')
    args = parser.parse_args()

    incremental_learning, verbose, run_shap_main, run_shap_inter, gpu = [
        parse_bool(x) for x in (args.incremental_learning, args.verbose, args.run_shap_main,
                                args.run_shap_inter, args.gpu)]

    interface, queue = [None if x.lower() == 'none' else x for x in (args.interface, args.worker_queue)]

    if gpu:
        tree_method = 'gpu_hist'
        raise NotImplementedError('GPUs not supported yet')
    else:
        tree_method = 'hist'

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

    t0 = time.time()
    if args.seed > 0:
        np.random.seed(args.seed)

    with spin_cluster(args.cluster, args.n_threads_per_worker, args.local_dir, 1, args.mem_per_worker,
                      args.time_per_worker, interface, queue) as cluster:
        scale_cluster(cluster, args.cluster, args.n_workers_in_cluster, args.n_threads_per_worker, args.mem_per_worker)
        with Client(cluster) as client:
            # print(f'Waiting to scale up to {args.n_workers_in_cluster} workers before continuing...')
            # client.wait_for_workers(args.n_workers_in_cluster, 300)

            print(f'Returned client with info/address: {client}')
            with h5py.File(args.in_ml, 'r') as f:
                X, y, rows, columns = read_ml(args.in_ml, f, row_chunks=args.row_chunk_size,
                                              x_key=args.xkey, y_key=args.ykey)
                y_binary = da.from_array(f['y'], chunks=(args.row_chunk_size, 1))
                column_names = columns.squeeze().to_numpy()

                fit_kwargs = dict(zip(['n_threads', 'eval_metric', 'loss', 'verbose', 'tree_method'],
                                      [args.n_threads_per_worker, eval_metric, loss, verbose, tree_method]))

                # make all subdirectories for files
                sub_dirs = [os.path.join(args.out, 'refit', z) for z in ['models', 'predictors', 'importances']]
                _ = [pathlib.Path(z).mkdir(parents=True, exist_ok=True) for z in sub_dirs]

                print(f'Waiting to scale up to {args.n_workers_in_cluster} workers before continuing...')
                client.wait_for_workers(n_workers=args.n_workers_in_cluster)

                model, y_pred, importance_scores, used_cols = main(
                    client, X, y, column_names, args.n_folds, incremental_learning,
                    args.row_chunk_size, args.out, args.prefix, args.incremental_start_round,
                    args.incremental_n_boost_per_round, run_shap_main, run_shap_inter, args.hp_search_results,
                    platt_scale, y_binary=y_binary, score_method=score_method,
                    n_booster_overide=args.n_booster_overide, **fit_kwargs
                )

    t1 = time.time()
    t2 = t1 - t0
    print(f'\nTime taken to process on CPU: '
          f'{t2 // 3600 % 24:.2f} hours, {t2 // 60 % 60:.2f} minutes, {t2 % 60:.2f} seconds')
