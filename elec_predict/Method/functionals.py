import numpy as np


def normalize(x, axis=-1, fillna=True, mask=None):
    if mask is None:
        mask = np.isnan(x)
    else:
        mask = np.bitwise_or(np.isnan(x), mask)
    x = np.ma.MaskedArray(x, mask=mask)
    mu = np.ma.mean(x, axis=axis, keepdims=True)
    std = np.ma.std(x, axis=axis, keepdims=True)
    x_norm = (x - mu) / (std + 1e-6)
    if fillna:
        x_norm = np.nan_to_num(x_norm)
    return x_norm.data, mu.data, std.data
