import numpy as np
import torch.utils.data as torch_data


class Seq2SeqSampler(torch_data.Sampler):

    def __init__(self, data_source, batch_size, sampling_rate=1., random_seed=42):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.random_seed = np.random.RandomState(random_seed)

    def __iter__(self):
        if self.data_source.num_series == 1:
            starts = np.arange(self.data_source.num_starts)
            self.random_seed.shuffle(starts)
            for i in range(len(self)):
                yield np.array([0]), starts[i*self.batch_size: (i+1)*self.batch_size]
        else:
            idxes = np.arange(self.data_source.num_series)
            starts = np.arange(self.data_source.num_starts)
            self.random_seed.shuffle(idxes)
            for i in range(len(self)):
                start = self.random_seed.choice(starts)
                yield idxes[i * self.batch_size: (i + 1) * self.batch_size], np.array([start])

    def __len__(self):
        n = np.floor(len(self.data_source) * self.sampling_rate)
        if n % self.batch_size == 0:
            return int(n // self.batch_size)
        else:
            return int(n // self.batch_size) + 1
