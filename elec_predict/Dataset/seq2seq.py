import torch
import numpy as np
from torch.utils.data import Dataset

class Seq2SeqDataset(Dataset):

    def __init__(self, series, enc_len, dec_len, features=None, weights=None, seq_last=True,
                 device='cpu', mode='train'):
        self.series = series
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.weights = weights
        self.seq_last = seq_last
        self.device = device
        self.mode = mode

        self.num_series = series.data.shape[0]
        self.num_starts = series.data.shape[2] - enc_len - dec_len + 1
        self.features = features
        self.enc_num = self.feature_filter(lambda x: x.enc and not x.is_cat, features)
        self.enc_cat = self.feature_filter(lambda x: x.enc and x.is_cat, features)
        self.dec_num = self.feature_filter(lambda x: x.dec and not x.is_cat, features)
        self.dec_cat = self.feature_filter(lambda x: x.dec and x.is_cat, features)

    @staticmethod
    def feature_filter(func, features):
        if features is None:
            return None
        ret = list(filter(func, features))
        if len(ret) == 0: return None
        return ret

    def __len__(self):

        if self.num_series == 1:
            return self.num_starts
        else:
            return self.num_series

    def read_batch(self, features, series_idx, time_idx):
        if features is None:
            return None
        batch = np.concatenate([f.read_batch(series_idx, time_idx) for f in features], axis=1)
        if not self.seq_last:
            batch = batch.transpose([0, 2, 1])
        return torch.as_tensor(batch, dtype=torch.long if features[0].is_cat else torch.float, device=self.device)

    def __getitem__(self, items):
        """

        Args:
            items: (series idxes: 1D array, time idxes: 2D array)
        Returns:

        """
        enc_idx = np.stack([np.arange(i, i+self.enc_len) for i in items[1]], axis=0)
        dec_idx = np.stack([np.arange(i+self.enc_len, i+self.enc_len+self.dec_len) for i in items[1]], axis=0)
        series_idx = items[0]

        feed_x = {
            "enc_x": self.read_batch([self.series], series_idx, enc_idx),
            "dec_x": self.read_batch([self.series], series_idx, dec_idx - 1),
            "enc_num": self.read_batch(self.enc_num, series_idx, enc_idx),
            "dec_num": self.read_batch(self.dec_num, series_idx, dec_idx),
            "enc_cat": self.read_batch(self.enc_cat, series_idx, enc_idx),
            "dec_cat": self.read_batch(self.dec_cat, series_idx, dec_idx),
            "dec_len": dec_idx.shape[1],
        }
        feed_y = self.read_batch([self.series], series_idx, dec_idx)
        weight = self.read_batch([self.weights], series_idx, dec_idx) if self.weights is not None else None
        return feed_x, feed_y, weight
