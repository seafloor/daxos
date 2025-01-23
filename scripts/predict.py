import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from daxos.read import read_ml, load_booster
from daxos.explain import collect_importances, subset_predictors
from daxos.utils import parse_bool, yhat, force_none_if_str_empty
from daxos.distribute import spin_cluster, scale_cluster
from daxos.crossvalidate import persist_daskdmatrix, score_model
from dask.distributed import Client
import dask.array as da
from daxos.deconfound import adjust_for_covars, read_betas, get_x_y_save_paths
import pathlib
import numpy as np
import joblib
import xgboost as xgb
import pandas as pd
import argparse
import h5py


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using Dask and XGBoost',
                                     epilog='Author: Matthew Bracher-Smith (smithmr5@cardiff.ac.uk)')
    parser.add_argument('--in_ml', type=str, default='',
                        help='.hdf5 file with "/x" and "/y" as da.arrays, and "/rows" and "/cols" as pd.DataFrames.')
    parser.add_argument('--used_cols', type=str, default='',
                        help='.csv file with used columns in to')
    parser.add_argument('--bst_path', type=str, default='',
                        help='.json file saved xgb model.')
    parser.add_argument('--cluster', type=str, default='local',
                        help='Run local cluster on current node or use as the root to spin-up a distributed cluster.')
    parser.add_argument('--n_threads_per_worker', type=int, default=5,
                        help='Number of threads used - passed to Dask and XGBoost for multi-threading.')
    parser.add_argument('--time_per_worker', type=str, default='05:00:00',
                        help='Time for compute nodes in dsitributed cluster - should be shorter than scheduler time.')
    parser.add_argument('--mem_per_worker', type=str, default='30GB',
                        help='Total amount of RAM per node (not per CPU). Will be same for all n_workers_in_cluster.')
    parser.add_argument('--n_workers_in_cluster', type=int, default=4,
                        help='Number of compute nodes in SlurmCluster. Passed to cluster.scale(n_workers_in_cluster).')
    parser.add_argument('--out', type=str,
                        help='Full path for directory to save output files.')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for output filename.')
    parser.add_argument('--local_dir', type=str, default='',
                        help='Local directory for dask workers.')
    parser.add_argument('--row_chunk_size', type=int, default=100,
                        help='Size of the chunks when reading in dask arrays.')
    parser.add_argument('--run_shap_main', type=str, default='False',
                        help='If true, generate SHAP values')
    parser.add_argument('--run_shap_inter', type=str, default='False',
                        help='If true, generate SHAP interaction values')
    parser.add_argument('--platt_scale_model', type=str, default=None,
                        help='Full path of fit logistic regression or other sklearn model. Will be loaded and passed '
                             'the test set predictions from XGBoost as input.')
    parser.add_argument('--gpu', type=str, default='False',
                        help='Use GPUs if True.')
    parser.add_argument('--gpu_resources', type=str, default='gpu:1',
                        help='Specify as <resource>:<type>:<count>, where type is optional. Passed to --gres=')
    parser.add_argument('--interface', type=str, default='None',
                        help='Networking interface for connection between workers. Uses "lo" if --cluster is "local"')
    parser.add_argument('--xkey', type=str, default='x',
                        help='Key in hdf5 file for X.')
    parser.add_argument('--ykey', type=str, default='y',
                        help='Key in hdf5 file for y.')
    parser.add_argument('--worker_queue', type=str, default='None',
                        help='Name of queue to submit worker jobs to. Default None.')
    args = parser.parse_args()

    shap_main, shap_inter, gpu = [parse_bool(x) for x in (args.run_shap_main, args.run_shap_inter, args.gpu)]
    interface, queue = [None if x.lower() == 'none' else x for x in (args.interface, args.worker_queue)]

    # load platt scaling model for platt scaling and set scoring method for model evaluation
    platt_scale_model = force_none_if_str_empty(args.platt_scale_model)

    if args.ykey == 'y_adjusted':
        assert platt_scale_model is not None, 'Platt model required for continuous y'
        assert all([os.path.exists(platt_scale_model) and os.path.getsize(platt_scale_model) > 0]), 'Platt model does not exist'
        score_method = 'RMSE'
        platt_model = joblib.load(platt_scale_model)
    else:
        score_method = 'AUC'
        platt_model = None

    print(f'\nSetting score_method as {score_method} because platt_scale_model is {platt_scale_model}\n')

    t0 = time.time()

    with spin_cluster(cluster_type=args.cluster, n_threads=args.n_threads_per_worker,local_dir=args.local_dir, processes=1, 
                      mem=args.mem_per_worker, walltime=args.time_per_worker, interface=interface, queue=queue, gpu=gpu,
                      gpu_resources=args.gpu_resources) as cluster:
        scale_cluster(cluster, args.cluster, args.n_workers_in_cluster, args.n_threads_per_worker, args.mem_per_worker)

        with Client(cluster) as client:
            print(f'Returned client with info/address: {client}')
            with h5py.File(args.in_ml, 'r') as f:
                X, y, rows, columns = read_ml(args.in_ml, f, row_chunks=args.row_chunk_size,
                                              x_key=args.xkey, y_key=args.ykey)
                y_binary = da.from_array(f['y'], chunks=(args.row_chunk_size, 1))
                colnames = columns.squeeze().to_numpy()

                print('\n--> Checking col names in X against those used in the model fit')
                used_cols = pd.read_csv(args.used_cols).iloc[:, 0].to_numpy()
                print(f'Loaded {len(used_cols)} column names from previous fit')
                print(f'Example columns: {used_cols[:5]}')

                if X.shape[1] != len(used_cols):
                    print("Number of cols in X and those used in last fit don't match")
                    X, out_cols = subset_predictors(X, colnames, subset_cols=used_cols)
                else:
                    print("Number of cols in X and those used in last fit match. Skipping predictor subsetting.")
                    out_cols = colnames

                print(f'\n--> Attemping to load XGB model from {args.bst_path}')
                # handling any accidental double slashes as file name made in shell script
                model_path = args.bst_path.replace('//', '/')

                if os.path.exists(model_path):
                    bst = load_booster(model_path)
                else:
                    FileNotFoundError(f'XGB model file {model_path} not found')

                dtest = persist_daskdmatrix(client, X, y, feature_names=out_cols, gpu=gpu)

                print('\n--> Predicting...')
                prediction = xgb.dask.predict(client, bst, dtest)
                test_set_score = score_model(y, prediction, score_method=score_method)
                print(f"{score_method} on test set (CAN be used to measure performance): {test_set_score:.2g}")

                if platt_model is None:
                    y_pred_platt_scaled = np.empty((y.shape[0],))
                    y_pred_platt_scaled[:] = np.nan
                else:
                    y_pred_platt_scaled = yhat(platt_model, prediction.reshape(-1, 1))
                    platt_scaled_score = score_model(y_binary, y_pred_platt_scaled, score_method='AUC')
                    print(f"Platt's AUC on test set (CAN be used to measure performance): {platt_scaled_score:.2g}")

                y_pred = pd.DataFrame({'IID': rows['IID'].to_numpy().squeeze(),
                                       'y_true': y_binary.squeeze(),
                                       'y_pred': prediction.squeeze(),
                                       'y_pred_platt_scaled': y_pred_platt_scaled})

                importance_outdir = os.path.join(args.out, 'predict', 'importances')
                y_pred_outdir = os.path.join(args.out, 'predict', 'predictions')
                _ = [pathlib.Path(x).mkdir(parents=True, exist_ok=True) for x in (importance_outdir, y_pred_outdir)]

                y_pred.to_csv(os.path.join(y_pred_outdir, f'{args.prefix}_y_pred.csv'), index=False)
                _ = collect_importances(client, bst, dtest, colnames=out_cols, out_dir=importance_outdir,
                                        out_prefix=args.prefix, run_shap_main=shap_main, run_shap_inter=shap_inter)

    t1 = time.time()
    t2 = t1 - t0
    print(f'\nTime taken to process on CPU: '
          f'{t2 // 3600 % 24:.2f} hours, {t2 // 60 % 60:.2f} minutes, {t2 % 60:.2f} seconds')
