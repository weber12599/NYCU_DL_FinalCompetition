import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SpeechRecognitionModel
from utils import AudioDataset, Data, train_data_processing, test_data_processing


class Model():
    def __init__(
        self,
        num_label: int,
        vocab_size: int
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = SpeechRecognitionModel(vocab_size=vocab_size, num_label=num_label).to(self.device)
        return

    def run_one_epoch(
        self,
        dataloader: DataLoader,
        ep_now: int,
        train: bool
    ) -> (float, float):
        if train:
            self.net.train()
        else:
            self.net.eval()
        Loss_CE, Loss_CTC, Correct, Total = 0, 0, 0, 0
        pbar = tqdm(dataloader)
        for it_now, (X_batch, S_batch, T_batch, input_lengths, label_lengths) in enumerate(pbar):
            # forward
            X_batch, S_batch, T_batch = X_batch.to(self.device), S_batch.to(self.device), T_batch.to(self.device)
            s_out, t_out = self.net(X_batch)

            # get loss
            loss_ce = self.loss_ce(s_out, S_batch)
            loss_ctc = self.loss_ctc(
                F.log_softmax(t_out, dim=2).transpose(0, 1), T_batch,
                input_lengths, label_lengths
            )
            loss = loss_ce + loss_ctc

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # log
            _, pred = torch.max(s_out.data, 1)
            total = S_batch.shape[0]
            correct = (pred == S_batch).sum().item()
            loss_ce = loss_ce.item()
            loss_ctc = loss_ctc.item()
            pbar.set_description('ep: %2d | %.3f, %.3f, %.3f' % (ep_now, loss_ce, loss_ctc, correct / total))
            Correct += correct
            Total += total
            Loss_CE += loss_ce * total
            Loss_CTC += loss_ctc * total

        self.writer.add_scalar('loss_ce/' + ('train' if train else 'valid'), Loss_CE / Total, ep_now)
        self.writer.add_scalar('loss_ctc/' + ('train' if train else 'valid'), Loss_CTC / Total, ep_now)
        self.writer.add_scalar('accuracy/' + ('train' if train else 'valid'), Correct / Total, ep_now)
        print('%5s  | ce: %.3f, ctc: %.3f, acc: %.3f' % (('train' if train else 'valid'), Loss_CE / Total, Loss_CTC / Total, Correct / Total))
        return

    def train_valid(
        self,
        train_dataset: AudioDataset,
        valid_dataset: AudioDataset,
        epoch: int,
        batch_size: int,
        leanring_rate: float,
        result_dir: str = 'result',
        exp_name: str = 'tmp'
    ) -> None:
        if epoch <= 0:
            return

        # setup loss function, optimizer
        self.loss_ce = nn.CrossEntropyLoss().to(self.device)
        self.loss_ctc = nn.CTCLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=leanring_rate)

        # setup tensorboard
        result_path = '%s/%s' % (result_dir, exp_name)
        if os.path.isdir(result_path):
            os.system('rm -r %s' % result_path)
            time.sleep(5)
        self.writer = SummaryWriter(result_path)

        # setup training, validation dataloader
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: train_data_processing(x)
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: train_data_processing(x)
        )

        # training loop
        for ep_now in range(epoch):
            self.run_one_epoch(train_dataloader, ep_now, True)
            with torch.no_grad():
                self.run_one_epoch(valid_dataloader, ep_now, False)
            self.save(flag=str(ep_now))
            print('---')
        return

    def test(
        self,
        dataset: AudioDataset,
        scenario_set: list,
        file_name: str
    ) -> None:
        # setup dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=lambda x: test_data_processing(x)
        )

        # get results
        results = []
        file_names = []
        with torch.no_grad():
            for _, (X_batch, file_name_batch) in enumerate(tqdm(dataloader)):
                X_batch = X_batch.to(self.device)
                s_out, t_out = self.net(X_batch)
                _, pred = torch.max(s_out.data, 1)
                results.extend(pred.detach().cpu().tolist())
                file_names.extend(file_name_batch)

        # write result into csv
        with open('./submission/%s.csv' % file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['File', 'Category'])
            for idx, result in enumerate(results):
                writer.writerow([idx, scenario_set[result]])
        return

    def save(
        self,
        file_name: str = './weight/tmp',
        flag: str = ''
    ) -> None:
        torch.save(self.net, file_name + flag + '.pt')
        return

    def load(
        self,
        file_name: str
    ) -> None:
        self.net = torch.load(file_name)
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7122)
    parser.add_argument("--model_path", type=str, default='./weight/tmp.pt')
    parser.add_argument("--submission", type=str, default='submission')
    parser.add_argument("--load", type=bool, default=False)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--leanring_rate", type=float, default=1e-4)
    args = parser.parse_args()

    # fix seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setup data
    data = Data(
        dataset_dir='./dataset',
        splits=(['train', 'test'] if args.epoch > 0 else ['test']),
        valid_ratio=args.valid_ratio
    )

    # setup model
    model = Model(
        num_label=data.num_label,
        vocab_size=len(data.vocab)
    )
    model.load(args.model_path) if args.load else None

    # train, valid
    model.train_valid(
        train_dataset=data.train,
        valid_dataset=data.valid,
        epoch=args.epoch,
        batch_size=args.batch_size,
        leanring_rate=args.leanring_rate
    )
    model.save()

    # test
    model.test(
        dataset=data.test,
        scenario_set=data.scenario_set,
        file_name=args.submission
    )
    return


if __name__ == '__main__':
    main()
