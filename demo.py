import torch
import numpy as np
import matplotlib.pyplot as plt

from elec_predict.Model.wave2wave import Wave2Wave
from elec_predict.Model.rnn2rnn import RNN2RNN
from elec_predict.Loss.rmse import RMSE
from elec_predict.Method.split import forward_split
from elec_predict.Method.functional import normalize
from elec_predict.Module.trainer import Trainer

from torch.utils.data import DataLoader

from elec_predict.Data.value import Value
from elec_predict.Dataset.seq2seq import Seq2SeqDataset
from elec_predict.Module.seq2seq_sampler import Seq2SeqSampler


def seq2seq_collate_fn(batch):
    (x, y, weight) = batch[0]
    return x, y, weight


def create_seq2seq_data_loader(series, enc_len, dec_len, batch_size, time_idx=None, weights=None, sampling_rate=1.,
                               features=None, seq_last=False, device='cpu', mode='train', seed=42,
                               num_workers=0, pin_memory=False):
    series = Value(series, 'series').sub(time_idx)
    weights = None if weights is None else Value(weights, 'weights').sub(time_idx)
    features = None if features is None else [f.sub(time_idx) for f in features]
    data_set = Seq2SeqDataset(series, enc_len, dec_len, features, weights, seq_last, device, mode)
    sampler = Seq2SeqSampler(data_set, batch_size, sampling_rate, seed)
    data_loader = DataLoader(data_set, collate_fn=seq2seq_collate_fn, sampler=sampler,
                                        num_workers=num_workers, pin_memory=pin_memory)
    # logger.info(f"---------- {mode} dataset information ----------")
    # logger.info(data_loader.dataset.info)
    # proportion = batch_size * num_iterations / len(data_set)
    # logger.info(f"data loader sampling proportion of each epoch: {proportion*100:.1f}%")
    return data_loader

batch_size = 16
enc_len = 36
dec_len = 12
series_len = 1000

epoch = 100
lr = 1e-3

valid_size = 12
test_size = 12

series = np.sin(np.arange(0, series_len)) + np.random.normal(0, 0.1, series_len) + np.log2(np.arange(1, series_len+1))
series = series.reshape(1, 1, -1)

train_idx, valid_idx = forward_split(np.arange(series_len), enc_len=enc_len, valid_size=valid_size+test_size)
valid_idx, test_idx = forward_split(valid_idx, enc_len, test_size)

# mask test, will not be used for calculating mean/std.
mask = np.zeros_like(series).astype(bool)
mask[:, :, test_idx] = False
series, mu, std = normalize(series, axis=2, fillna=True, mask=mask)

# create train/valid dataset
train_dl = create_seq2seq_data_loader(series[:, :, train_idx], enc_len, dec_len, sampling_rate=0.1,
                                      batch_size=batch_size, seq_last=True, device='cuda')
valid_dl = create_seq2seq_data_loader(series[:, :, valid_idx], enc_len, dec_len,
                                      batch_size=batch_size, seq_last=True, device='cuda')

# define model
wave = Wave2Wave(target_size=1, num_layers=6, num_blocks=1, dropout=0.1, loss_fn=RMSE())
wave.cuda()
opt = torch.optim.Adam(wave.parameters(), lr=lr)

# train model
wave_learner = Trainer(wave, opt)
wave_learner.fit(max_epochs=epoch, train_dl=train_dl, valid_dl=valid_dl, early_stopping=True, patient=16)

# load best model
wave_learner.load(wave_learner.best_epoch)

# predict and show result
wave_preds = wave_learner.model.predict(torch.tensor(series[:, :, test_idx[:-12]]).float().cuda(), 12).cpu().numpy().reshape(-1)

plt.plot(series[:, :, -48:-12].reshape(-1))
plt.plot(np.arange(36, 48), wave_preds, label="wave2wave preds")
plt.plot(np.arange(36, 48), series[:, :, test_idx[-12:]].reshape(-1), label="target")
plt.legend()
