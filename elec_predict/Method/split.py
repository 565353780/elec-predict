import numpy as np


def forward_split(time_idx, enc_len, valid_size):
    if valid_size < 1:
        valid_size = int(np.floor(len(time_idx) * valid_size))
    valid_idx = time_idx[-(valid_size + enc_len):]
    train_idx = time_idx[:-valid_size]
    return train_idx, valid_idx


