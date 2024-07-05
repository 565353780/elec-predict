import numpy as np


class Value:

    def __init__(self, data, name, enc=True, dec=True, mapping=None):
        assert isinstance(data, np.ndarray)
        assert data.ndim in [2, 3]
        self.is_property = False
        if data.ndim == 2:
            self.is_property = True
            data = np.expand_dims(data, axis=2)
        self.data = data
        self.name = name
        self.enc = enc
        self.dec = dec
        self.is_cat = True if str(self.data.dtype)[:3] == "int" else False
        self.mapping = mapping

    def sub(self, time_idx=None):
        if time_idx is None:
            return self
        if self.is_property:
            return self
        else:
            return Value(self.data[:, :, time_idx], self.name, self.enc, self.dec, self.mapping)

    def read_batch(self, series_idx, time_idx):
        """

        Args:
            series_idx: 1D array
            time_idx: 2D array

        Returns:

        """
        if self.mapping is not None:
            series_idx = np.array([self.mapping.get(i) for i in series_idx])

        if self.is_property:
            if len(series_idx) == 1:
                batch = self.data[[series_idx.item()]].repeat(time_idx.shape[1], axis=2).repeat(time_idx.shape[0], axis=0)
            else:
                batch = self.data[series_idx].repeat(time_idx.shape[1], axis=2)
            return batch

        if len(series_idx) == 1:
            batch = self.data[series_idx.item()][:, time_idx].transpose(1, 0, 2)
        else:
            batch = self.data[series_idx, :, :][:, :, time_idx.squeeze()]
        return batch


