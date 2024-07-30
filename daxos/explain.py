import dask.array as da
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import shutil
from .crossvalidate import persist_daskdmatrix, fit_dask_xgb


def subset_after_shap(used_cols, all_cols, shap_values, value_type='marginal', drop_bias=True):
    col_bool = np.hstack([np.isin(all_cols, used_cols), [not drop_bias]])

    if value_type == 'marginal':
        return shap_values[:, col_bool]
    elif value_type == 'interaction':
        return shap_values[:, col_bool, :][:, :, col_bool]
    else:
        raise ValueError('SHAP value type must be "marginal" or "interaction"')


def subset_predictors(X, all_cols, subset_cols):
    print(f'\n--> Subsetting predictors from {len(all_cols)} to {len(subset_cols)}')
    col_bool = np.isin(all_cols, subset_cols)

    print(f'Shape before subsetting {X.shape}')
    X = X[:, col_bool]
    out_cols = np.array(all_cols)[col_bool]

    print(f'Shape after subsetting {X.shape}')

    return X, out_cols


def subset_predictors_and_refit(client, X, y, used_cols, all_cols, params, **fit_kwargs):
    X, out_cols = subset_predictors(X, all_cols, subset_cols=used_cols)

    print('\nCreating reduced DMatrix for XGB and SHAP')
    dtrain = persist_daskdmatrix(client, X, y, feature_names=out_cols)

    print('\nTraining on reduced DMatrix...')
    bst, history = fit_dask_xgb(client, data=dtrain, params=params, **fit_kwargs)

    return dtrain, bst, out_cols


def collect_importances(client, bst, dtrain, colnames, out_dir, out_prefix, run_shap_main=False,
                        run_shap_inter=False):
    print('\n--> Collecting classic importance scores...')
    importance = (pd.DataFrame({'gain': pd.Series(bst.get_score(importance_type='gain')),
                                'weight': pd.Series(bst.get_score(importance_type='weight')),
                                'cover': pd.Series(bst.get_score(importance_type='cover')),
                                'total_gain': pd.Series(bst.get_score(importance_type='total_gain')),
                                'total_cover': pd.Series(bst.get_score(importance_type='total_cover'))})
                    .reset_index())
    importance.columns = ['predictors', 'gain', 'weight', 'cover', 'total_gain', 'total_cover']
    importance.to_csv(os.path.join(out_dir, out_prefix + '_' + 'imp.csv'), index=False)

    print('Top predictors, sorted by gain:')
    print(importance.sort_values('gain', ascending=False).head())
    print('Collected classic importance scores for {} predictors'.format(importance.shape[0]))

    if run_shap_main:
        print('\n--> Collecting SHAP values...')
        print('Subsetting to only the {} predictors used to build trees for all SHAP values'.format(importance.shape[0]))
        shap_path = os.path.join(out_dir, out_prefix + '_' + 'shap_main.zarr')
        shap_colnames_path = os.path.join(out_dir, out_prefix + '_' + 'shap_colnames.csv')
        if os.path.exists(shap_path):
            print('Overwriting previously saved SHAP values in: {}'.format(shap_path))
            shutil.rmtree(shap_path, ignore_errors=True)

        shap_values = xgb.dask.predict(client, bst, dtrain, pred_contribs=True)
        shap_values = subset_after_shap(importance['predictors'].values, colnames, shap_values,
                                        value_type='marginal', drop_bias=True)
        print('Obtained SHAP values of shape: {}'.format(shap_values.shape))
        da.to_zarr(shap_values, shap_path)
        importance['shap_mean_abs'] = np.mean(np.abs(shap_values), axis=0)
        print('Top predictors, sorted by mean(|SHAP|) values:')
        print(importance.sort_values('shap_mean_abs', ascending=False).head())
        importance.to_csv(os.path.join(out_dir, out_prefix + '_' + 'imp.csv'), index=False)

        print('\nAdding mean(|SHAP|) values to saved importance scores')
        importance.predictors.to_csv(shap_colnames_path, index=False, header=None)

        print('\nSaving the column names for SHAP value files (only use for loading in SHAP main/interaction effects)')
        if os.path.exists(shap_colnames_path):
            print(f'Saved to path: {shap_colnames_path}')

    if run_shap_inter:
        print('\n--> Collecting SHAP interaction values...')
        shap_inter_path = os.path.join(out_dir, out_prefix + '_' + 'shap_interaction.zarr')
        if os.path.exists(shap_inter_path):
            print('Overwriting previously saved SHAP values in: {}'.format(shap_inter_path))
            shutil.rmtree(shap_inter_path, ignore_errors=True)

        shap_interaction_values = xgb.dask.predict(client, bst, dtrain, pred_interactions=True)
        shap_interaction_values = subset_after_shap(importance['predictors'].values, colnames, shap_interaction_values,
                                                    value_type='interaction', drop_bias=True)
        print('Obtained SHAP interaction values of shape: {}'.format(shap_interaction_values.shape))
        da.to_zarr(shap_interaction_values, shap_inter_path)

    return importance
