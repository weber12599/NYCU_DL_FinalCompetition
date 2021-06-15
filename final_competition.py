import os
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torchaudio
import numpy as np


class Dataset():
    def __init__(
        self,
        dataset_dir: str,
        splits: list,
        valid_ratio: float,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        n_mels: int = 64
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.train, self.valid, self.test = None, None, None

        def transform_to_log_mel(
            split: str,
            file_names: list,
        ) -> (list, list):
            datas, times = [], []
            for file_name in tqdm(file_names):
                waveform, sample_rate = torchaudio.load('%s/%s/%s' % (dataset_dir, split, file_name))
                datas.append(self.mel_transform(waveform).log2())
                times.append(datas[-1].shape[-1])
            return datas, times

        for split in splits:
            file_names = os.listdir('%s/%s' % (dataset_dir, split))
            print('%s: %d' % (split, len(file_names)))
            if split == 'train':
                random.shuffle(file_names)
                train_end_idx = int(len(file_names) * (1 - valid_ratio))
                self.train, times_train = transform_to_log_mel(split=split, file_names=file_names[:train_end_idx])
                self.valid, times_valid = transform_to_log_mel(split=split, file_names=file_names[train_end_idx:])
            elif split == 'test':
                self.test, times_test = transform_to_log_mel(split=split, file_names=file_names)
            else:
                raise NotImplementedError
        # times = times_train + times_valid + times_test
        # print(np.average(times), np.std(times), np.median(times), np.min(times), np.max(times))
        return


class Net(nn.Module):
    def __init__(
        self
    ) -> None:
        super(Net, self).__init__()
        return

    def forward(
        self
    ) -> torch.Tensor:
        return


class Model():
    def __init__(
        self,
        net: Net,
        epoch: int,
        batch_size: int
    ) -> None:
        super().__init__()
        self.epoch, self.batch_size = epoch, batch_size
        self.net = net
        return

    def run_one_epoch(
        self,
        datas: list
    ) -> (float, float):
        iteration = int(np.ceil(len(datas) / self.batch_size))
        for it_now in range(iteration):
            data = datas[it_now * self.batch_size: (it_now + 1) * self.batch_size]
            # pass into net
            # get loss
            # update parameters
        return random.uniform(-100, 100), random.uniform(0, 100)

    def train_valid(
        self,
        train_datas: list,
        valid_datas: list,
        result_dir: str = 'result',
        exp_name: str = 'tmp'
    ) -> None:
        writer = SummaryWriter('%s/%s' % (result_dir, exp_name))
        for ep_now in range(self.epoch):
            train_loss, train_acc = self.run_one_epoch(train_datas)
            valid_loss, valid_acc = self.run_one_epoch(valid_datas)
            # log
            writer.add_scalar('loss/train', train_loss, ep_now)
            writer.add_scalar('loss/valid', valid_loss, ep_now)
            writer.add_scalar('accuracy/train', train_acc, ep_now)
            writer.add_scalar('accuracy/valid', valid_acc, ep_now)
        return

    def test(
        self,
        datas: list
    ) -> list:
        results = []
        # get results
        return results


def main():
    seed = 7122
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # setup data
    dataset = Dataset(dataset_dir='./dataset', splits=['train', 'test'], valid_ratio=0.2)

    # setup model
    net = Net()
    model = Model(net, epoch=1000, batch_size=128)

    # train, valid, test
    model.train_valid(train_datas=dataset.train, valid_datas=dataset.valid)
    results = model.test(dataset.test)
    return


if __name__ == '__main__':
    main()
