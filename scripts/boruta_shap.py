import h5py
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'daxos'))
from daxos.boruta import boruta
from daxos.distribute import spin_cluster, scale_cluster
from daxos.crossvalidate import read_hp_search_results
from daxos.explain import subset_predictors
from daxos.read import read_ml
from dask.distributed import LocalCluster, Client


if __name__ == '__main__':
    print('started script')
    max_iter = 30
    percent = 100
    two_step=True

    cluster_type = 'distributed'
    n_threads_per_worker = 5
    n_workers_in_cluster = 6
    local_dir = '/workdir/matthew.smith'
    mem_per_worker = '64GB'
    time_per_worker = '36_00_00'
    interface = None
    queue = None
    datadir = '/workdir/matthew.smith/xgboost/USE_job_44274458_sex_adjusted_xy_no_center_may_2023_float16'

    in_ml = '/workdir/matthew.smith/data_freeze/plink_files/maf_0_05/clumped_r2_0_75/train/plink/' \
            'with_apoe_train_shuffled_sex_adjusted_NO_CENTER_may_2023_float16.hdf5'

    print('Spinning up local cluster on scheduler node to handle initial data loading')
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            with h5py.File(in_ml, 'r') as f:
                X, y, rows, columns = read_ml(in_ml, f, x_key='x_adjusted', y_key='y_adjusted')
                X = X.compute()
                y = y.compute()

    print('Dask arrays loaded into memory on scheduler')

    print('\nSpinning up distributed cluster to handling XGBoost training in boruta')
    with spin_cluster(cluster_type, n_threads_per_worker, local_dir, 1, mem_per_worker,
                      time_per_worker, interface, queue) as cluster:
        scale_cluster(cluster, cluster_type, n_workers_in_cluster, n_threads_per_worker, mem_per_worker)
        with Client(cluster) as client:
            #### subset to predictors in the xgb model ####
            used_cols = pd.read_csv(
                os.path.join(datadir, 'refit/predictors/sex_adjusted_xy_no_center_may_2023'
                                      '_float16_train_refit_used_cols.csv')).iloc[:, 0].to_numpy()
            colnames = columns.squeeze().to_numpy()

            if X.shape[1] != len(used_cols):
                print("Number of cols in X and those used in last fit don't match")
                X, out_cols = subset_predictors(X, colnames, subset_cols=used_cols)
            else:
                print("Number of cols in X and those used in last fit match. Skipping predictor subsetting.")
                out_cols = colnames

            # drop the _ATCG from rsids
            used_cols_simplified = pd.Series(out_cols).str.split('_', expand=True).iloc[:, 0].values

            # read best params
            best_params = read_hp_search_results(
                '/workdir/matthew.smith/xgboost/cv/clumped_075_maf_005_with_apoe/hpsearch')

            fit = {'n_threads': -1, 'eval_metric': 'rmse', 'loss': 'reg:squarederror',
                   'tree_method': 'hist'}

            client.wait_for_workers(n_workers=n_workers_in_cluster)

            important, tentative = boruta(X, y, [f'snp{i}' for i in range(1, X.shape[1] + 1)], best_params,
                                          client, importance='shap', shap_sumstat='mean', max_iter=max_iter,
                                          perc=percent, alpha=0.05, two_step=two_step, row_chunks=1000,
                                          train_split=1, train_or_test='train', **fit)

            print(f'{np.sum(important)} important predictors found')
            print(f'{np.sum(tentative)} tentative predictors found')

            out_step = 'two_step' if two_step else 'one_step'

            for l, n in zip([important, tentative], ['important', 'tentative']):
                if np.sum(l) > 0:
                    pd.Series(used_cols_simplified[l]).to_csv(
                        os.path.join(datadir, f'boruta/{n}_rsids_{out_step}_{percent}_perc_{max_iter}_iter.txt'), header=None, index=False)
