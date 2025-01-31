import numpy as np
from itertools import compress


def parse_bool(bool_arg):
    bool_arg = bool_arg.lower()
    if bool_arg == 'false':
        bool_arg = False
    elif bool_arg == 'true':
        bool_arg = True
    else:
        raise ValueError(f'Arg {bool_arg} not recognised. Must be in ["True", "False"].')

    return bool_arg


def force_none_if_str_empty(arg):
    if arg is None:
        return None
    elif isinstance(arg, str):
        if arg:
            if arg.lower() == 'none':
                return None
            else:
                return arg
        else:
            return None
    else:
        return arg


def set_numpy_dtype(dtype_str):
    dtype_options = [np.float16, np.float32, np.float64]
    dtype_numpy = list(compress(dtype_options, [dtype_str.lower() == s for s in ('float16', 'float32', 'float64')]))
    if len(dtype_numpy) != 1:
        raise ValueError(f'NumPy dtypes selected as {dtype_numpy}')

    return dtype_numpy[0]


def yhat(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        return clf.decision_function(X)
    elif hasattr(clf, 'predict'):
        return clf.predict(X)
    else:
        raise ValueError('clf has no predict attr')


def validate_pandas_columns():
    # raise ValueError
    pass


def validate_pandas_index():
    # raise ValueError
    pass


def validate_shape(data, nrows, ncols):
    if data.shape != (nrows, ncols):
        raise ValueError(f'Data expected to be shape {data.shape} but is shape {(nrows, ncols)}')


def validate_numpy_data_types():
    # raise TypeError
    pass


def validate_type():
    # raise TypeError
    pass
