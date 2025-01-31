from .read import load_booster
from dask.core import flatten
import dask.array as da
import scipy.stats as sp
from dask.distributed import wait
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, mean_squared_error
from glob import glob
import pathlib
import numpy as np
import xgboost as xgb
from pprint import pprint
import pandas as pd
import time
import dask
import math
import gc
import os


def map_partitions_to_workers(workers, partitions, splits):
    if len(workers) != len(splits):
        raise ValueError('Expected equal number of workers and splits')

    mapping = {}
    for wkey, split in zip(workers, splits):
        for idx in split:
            mapping[partitions[idx]] = wkey

    return mapping


def map_x_y_partitions_to_workers(client, X, y, verbose=True):
    # get keys for workers and each partition
    worker_keys = list(client.scheduler_info()['workers'].keys())
    partition_keys_X = list(flatten(X.__dask_keys__()))
    partition_keys_y = list(flatten(y.__dask_keys__()))

    # get an even splitting of partitions given the number of workers
    partition_splitting = np.array_split(range(X.npartitions), len(worker_keys))

    # create a partition:worker mapping of keys for X and y and combine
    X_mapping = map_partitions_to_workers(worker_keys, partition_keys_X, partition_splitting)
    y_mapping = map_partitions_to_workers(worker_keys, partition_keys_y, partition_splitting)
    mapping = {**X_mapping, **y_mapping}

    print(f'\n--> Manually allocating {X.npartitions} partitions to {len(worker_keys)} workers')

    if verbose:
        print('Mapping allocated:')
        pprint(mapping)

    return mapping


def persist_daskdmatrix(client, X, y, feature_names=None, manually_map_to_workers=True, method='persist', gpu=True):
    if feature_names is not None:
        assert len(feature_names) == X.shape[1], \
            f'len(feature_names) of {len(feature_names)} does not match no columns in X of {X.shape}'
        
        if gpu:
            feature_names = feature_names.tolist()
    
    if gpu:
        print('Persisting for GPUs - forcing dask worker scheduling rather than manually mapping')
        manually_map_to_workers = False
        # worker_info = client.scheduler_info()['workers']
        # gpu_workers = [w for w in worker_info.values() if 'gpu' in w.get('resources', {})]
        # if not gpu_workers:
        #     raise ValueError("GPU training requested but no GPU workers found")

    if manually_map_to_workers:
        mapping = map_x_y_partitions_to_workers(client, X, y)

        def key_to_worker(key):
            return mapping.get(key)

        with dask.annotate(workers=key_to_worker):
            if method == 'persist':
                X.persist()
                y.persist()
            elif method == 'scatter':
                X = client.scatter(X)
                y = client.scatter(y)

            wait(X)
            wait(y)

            if gpu:
                dtrain = xgb.dask.DaskQuantileDMatrix(client, X, y, feature_names=feature_names)
            else:
                dtrain = xgb.dask.DaskDMatrix(client, X, y, feature_names=feature_names)
    else:
        if method == 'persist':
            X.persist()
            y.persist()
        elif method == 'scatter':
            X = client.scatter(X)
            y = client.scatter(y)

        wait(X)
        wait(y)

        if gpu:
            dtrain = xgb.dask.DaskQuantileDMatrix(client, X, y, feature_names=feature_names)
        else:
            dtrain = xgb.dask.DaskDMatrix(client, X, y, feature_names=feature_names)

    return dtrain


def set_random_search_distributions(n_iter=30, subsample_min=0.5, subsample_max=1.0):
    params = dict(eta=sp.reciprocal(0.0001, 0.1),
                  colsample_bytree=sp.uniform(loc=0.5, scale=0.5),
                  max_depth=sp.randint(2, 9))

    if any([not math.isclose(subsample_min, 0.5), not math.isclose(subsample_max, 1.0)]):
        params['subsample'] = sp.uniform(loc=subsample_min, scale=subsample_max-subsample_min)
    else:
        params['subsample'] = sp.uniform(loc=0.5, scale=0.5)

    return ParameterSampler(params, n_iter)


def flatten_if_2d(a):

    return a if a.ndim == 1 else a.flatten()


def score_model(y_true, y_pred, score_method='AUC'):
    y_true, y_pred = [flatten_if_2d(y) for y in (y_true, y_pred)]

    # handle missing values in y_true which may result after adjusting it for covariates (or poor filtering/qc)
    y_true, y_pred = [y.compute() if hasattr(y, 'chunks') else y for y in (y_true, y_pred)]
    if np.isnan(y_true).sum() > 0:
        not_missing_bool = ~np.isnan(y_true)
        y_true = y_true[not_missing_bool]
        y_pred = y_pred[not_missing_bool]

    if score_method == 'AUC':
        return roc_auc_score(y_true, y_pred)
    elif score_method == 'RMSE':
        return math.sqrt(mean_squared_error(y_true, y_pred))
    else:
        raise ValueError(f'score_method {score_method} not recognised')


def fit_dask_xgb(client, data, params, xgb_model=None, n_threads=1, eval_metric='logloss', loss='binary:logistic',
                 verbose=False, tree_method='gpu_hist', gpu=True):
    
    xgb_args = {
        'verbosity': int(verbose),
        'tree_method': tree_method,
        'single_precision_histogram': True,
        'eval_metric': eval_metric,
        'objective': loss,
        'max_depth': params['max_depth'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'eta': params['eta']
    }

    output = xgb.dask.train(client,
                            xgb_args,
                            data,
                            num_boost_round=params['n_boost_round'],
                            xgb_model=xgb_model,
                            evals=[(data, 'train')])

    bst = output['booster']
    history = output['history']
    if verbose:
        print('Evaluation history:', history)

    return bst, history


def fit_one_round_cv(client, X, y, params, n_fold=5, score_method='AUC', colnames=None,
                     manually_map_to_workers=False, gpu=True, **fit_kwargs):
    folds = chunked_kfold_split(X, y, n_splits=n_fold)
    scores, y_pred, y_true = [], [], []
    for i, f in enumerate(folds):
        X_train, y_train = X[f[0], :], y[f[0], :]
        X_test, y_test = X[f[1], :], y[f[1], :]

        print(f'Creating Dask DMatrices in CV fold number {i}')
        dtrain = persist_daskdmatrix(client, X_train, y_train, feature_names=colnames,
                                     manually_map_to_workers=manually_map_to_workers,
                                     gpu=gpu)
        _ = persist_daskdmatrix(client, X_test, y_test, feature_names=colnames, gpu=gpu)

        bst, _ = fit_dask_xgb(client, data=dtrain, params=params, gpu=gpu, **fit_kwargs)

        test_pred = xgb.dask.predict(client, bst, X_test)

        scores.append(score_model(y_test, test_pred, score_method=score_method))
        y_pred.append(test_pred.squeeze())
        y_true.append(y_test.squeeze())

    return scores, y_pred, y_true


def incremental_fit_xgb(client, X, y, colnames, best_params, start_round, out_dir, out_prefix, n_boost_per_round=10,
                        row_chunks=100, gpu=True, **fit_kwargs):
    print('\n--> Starting incremental learning')
    read_subsample = best_params['subsample']

    if not math.isclose(best_params['subsample'], 1.0):
        print(f'Reading in {read_subsample} subsample of data to avoid memory overload')
        print('Setting in-memory subsampling in XGBoost fit to 1 to avoid subsampling twice')
        best_params['subsample'] = 1.0

    model_path = os.path.join(out_dir, 'increments', out_prefix + '_' + 'xgbmodel.json')
    print(f'Saving model to {model_path} after each boosting round')
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    if all([os.path.exists(model_path), start_round == 1]):
        print('Deleting pre-existing saved model before starting incremental learning')
        os.remove(model_path)
        bst = None
    elif all([os.path.exists(model_path), start_round > 1]):
        print('Pre-existing saved model found - loading booster...')
        bst = load_booster(model_path)
    else:
        bst = None

    bst_range = range(start_round, best_params['n_boost_round'] + 1, n_boost_per_round)
    for i, ith_round in enumerate(bst_range):
        print(f'Starting boosting iteration {i + 1} of {len(bst_range)}, building {n_boost_per_round} trees per round')
        X_refit, _, y_refit, _ = chunked_train_test_split(X, y, row_chunks, read_subsample)

        dtrain = persist_daskdmatrix(client, X_refit, y_refit, feature_names=colnames, gpu=gpu)
        bst, history = fit_dask_xgb(client, data=dtrain, params=best_params, xgb_model=bst, gpu=gpu, **fit_kwargs)
        print(f"Number of trees now in booster: {len(bst.trees_to_dataframe().Tree.unique())}")

        bst.save_model(model_path)

        del dtrain

    return dtrain, bst, history, best_params


def xgb_dask_cv(client, X, y, params, n_fold=5, colnames=None, boost_rounds=1000, score_method='AUC', **fit_kwargs):
    folds = chunked_kfold_split(X, y, n_splits=n_fold)
    if not isinstance(params, list):
        params = [params]  # necessary otherwise will get different params in each fold

    # iterate on folds first, not params, so that we only have to create each DaskDMatrix once
    c_ypred, cv_results = [], []
    for i, f in enumerate(folds):
        X_train, y_train = X[f[0], :], y[f[0], :]
        X_test, y_test = X[f[1], :], y[f[1], :]

        print(f'Creating Dask DMatrices in CV fold number {i}')
        dtrain = persist_daskdmatrix(client, X_train, y_train, feature_names=colnames, gpu=gpu)
        dtest = persist_daskdmatrix(client, X_test, y_test, feature_names=colnames, gpu=gpu)

        print(f'Running HP search in CV for {len(params)} params')
        for iter, p in enumerate(params):
            p['n_boost_round'] = boost_rounds
            print(f'Running iteration {iter + 1} with params: {p}')
            bst, history = fit_dask_xgb(client, data=dtrain, params=p, **fit_kwargs)

            t0_predict = time.time()
            train_pred = xgb.dask.predict(client, bst, X_train)
            test_pred = xgb.dask.predict(client, bst, X_test)
            t1_predict = time.time()
            t2_predict = t1_predict - t0_predict
            print('\nTime taken to predict: {:.2f} hours, {:.2f} minutes, {:.2f} seconds'.format(
                t2_predict // 3600 % 24, t2_predict // 60 % 60, t2_predict % 60))

            c_ypred.append(test_pred)

            params_df = (pd.DataFrame(p, index=[0])
                           .assign(score=score_model(y_test, test_pred, score_method=score_method),
                                   metric=score_method,
                                   n_boost_round=p['n_boost_round'],
                                   train_score=score_model(y_train, train_pred, score_method=score_method),
                                   fold=i))

            cv_results.append(params_df)

        del dtrain
        del dtest
        gc.collect()

    cv_results = pd.concat(cv_results, ignore_index=True).reset_index(drop=True)
    sort_ascending = score_method != 'AUC'
    reduced_cv = (cv_results.groupby(['eta', 'subsample', 'colsample_bytree', 'max_depth', 'metric'])
                            .mean()
                            .reset_index(drop=False)
                            .sort_values('score', ascending=sort_ascending))

    best_params, best_score = get_best_cv(reduced_cv, score_method=score_method, verbose=False)
    best_idx = cv_results.loc[(np.isclose(cv_results['eta'], best_params['eta'])) &
                              (np.isclose(cv_results['subsample'], best_params['subsample'])) &
                              (np.isclose(cv_results['colsample_bytree'], best_params['colsample_bytree'])) &
                              (cv_results['max_depth'] == best_params['max_depth']), :].index

    cv_y_pred = pd.DataFrame({'y_idx': np.hstack([f[1] for f in folds]),
                              'y_true': y.squeeze(),
                              'y_pred': np.hstack([c_ypred[i] for i in best_idx]),
                              'fold': np.hstack([np.repeat(i, f[1].shape[0]) for i, f in enumerate(folds)])})

    return reduced_cv, cv_y_pred


def get_best_cv(cv_scores, score_method='AUC', verbose=True):
    best_run = cv_scores.iloc[0, :]
    best_run_score = best_run['score']
    best_run_params = best_run[["max_depth", "subsample", "colsample_bytree", "eta"]].to_dict()
    best_run_params['max_depth'] = int(best_run_params['max_depth'])
    if verbose:
        print(f"Best CV achieved {score_method} of {best_run_score} using params {best_run_params}")

    return best_run_params, best_run_score


def read_hp_search_results(hp_search_file):
    if any([os.path.isfile(hp_search_file), os.path.isdir(hp_search_file)]):
        if os.path.isdir(hp_search_file):
            print('\n--> HP search results directory found')
            print('Reading in and merging all .CSV files in directory')
            hp_files = sorted(glob(os.path.join(hp_search_file, '*.csv')))
            hp_search_values = pd.concat([pd.read_csv(f) for f in hp_files], axis=0)
        elif os.path.isfile(hp_search_file):
            print('\n--> HP search results file found')
            hp_search_values = pd.read_csv(hp_search_file)
        else:
            raise ValueError(f'Hyperparameter path not file or dir: {hp_search_file}')

        # set sorting of CV results to be descending only if using AUC, else ascending
        score_method = hp_search_values['metric'].iat[0]
        sort_ascending = score_method != 'AUC'
        hp_search_values = (hp_search_values.sort_values('score', ascending=sort_ascending)
                                            .reset_index(drop=True))

        print('Top HP combinations from search:\n')
        print(hp_search_values.head())

        hp_search_values = hp_search_values.iloc[0, :].to_dict()
        int_cols = ['max_depth', 'n_boost_round']
        best_params = {key: (int(value) if key in int_cols else value) for key, value in hp_search_values.items()}
    else:
        raise FileNotFoundError('No HP search results file/directory found')

    return best_params


def chunked_train_test_split(X, y, row_chunks='auto', train_size=0.8):
    row_chunks = X.chunksize[0] if row_chunks == 'auto' else row_chunks
    assert all([train_size >= 0.1, train_size <= 0.9]), 'Training fraction must be between 0.1 and 0.9, inclusive.'

    idx = np.arange(len(X.chunks[0]) - 1)
    idx_choice = np.random.choice(idx, int((train_size * X.shape[0]) // row_chunks), replace=False)
    train_idx = np.hstack([np.arange(x * row_chunks, (x + 1) * row_chunks) for x in idx_choice])
    test_idx = np.arange(X.shape[0])[np.isin(np.arange(X.shape[0]), train_idx, invert=True)]

    print(f'Train fraction after adhering to chunk sizes: {len(train_idx)/X.shape[0]:.2g}')

    return X[train_idx, :], X[test_idx, :], y[train_idx, :], y[test_idx, :]


def chunked_kfold_split(X, y, n_splits, row_chunks='auto', dask=False, verbose=True):
    """
    implement chunk-based train-test split and CV for dask
    prevents additional splitting of chunks during CV
    shuffling deliberately not allowed because it would introduce the issue we're resolving
    """

    row_chunks = X.chunksize[0] if row_chunks == 'auto' else row_chunks

    single_chunk = (X.shape[0] // (row_chunks * n_splits)) * row_chunks
    fold_sizes = np.hstack([np.repeat(single_chunk, n_splits-1),
                            np.array([single_chunk + X.shape[0] - single_chunk * n_splits])])
    assert X.shape[0] == np.sum(fold_sizes)

    offset = 0
    full_array = np.arange(0, X.shape[0])
    kfolds = []
    for fold in fold_sizes:
        y_ = np.arange(0 + offset, fold + offset)
        full_array = np.arange(0, X.shape[0])
        X_ = full_array[np.isin(full_array, y_, assume_unique=True, invert=True)]

        if dask:
            kfolds.append((da.from_array(X_, chunks=row_chunks), da.from_array(y_, chunks=row_chunks)))
        else:
            kfolds.append((X_, y_))
        offset += fold

    if verbose:
        print('Chunked kfold splits created')

    return kfolds


def cv_xgb(client, X, y, cv_subsample, n_folds, n_iter_search, boost_rounds=1000, min_subsample=0.7, max_subsample=0.7,
           score_method='AUC', gpu=True, **fit_kwargs):
    if cv_subsample > 0:
        X_train, y_train = X[:cv_subsample, :], y[:cv_subsample, :]
        print('\n--> Downsampled dataset for HP tuning', X_train, y_train)
    else:
        X_train, y_train = X, y

    print(f'\n--> Running {n_folds}-fold cross-validation with random search')
    params = set_random_search_distributions(n_iter_search, min_subsample, max_subsample)
    scores, y_pred = xgb_dask_cv(client, X_train, y_train, params, n_fold=n_folds, boost_rounds=boost_rounds,
                                 score_method=score_method, gpu=gpu, **fit_kwargs)
    best_params, best_score = get_best_cv(scores)

    return best_params, best_score, scores, y_pred, params
