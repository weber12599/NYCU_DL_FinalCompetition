import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(
        self,
        mel_specs: list,
        scenarios: list or None,
        transcriptions: list or None,
        file_names: list
    ) -> None:
        self.mel_specs = mel_specs
        self.scenarios = scenarios
        self.transcriptions = transcriptions
        self.file_names = file_names
        return

    def __len__(
        self
    ) -> int:
        return len(self.mel_specs)

    def __getitem__(
        self,
        index: int
    ) -> (torch.Tensor, torch.Tensor or None, torch.Tensor or None):
        if self.scenarios is not None and self.transcriptions is not None:
            return (self.mel_specs[index], self.scenarios[index], self.transcriptions[index], self.file_names[index])
        else:
            return (self.mel_specs[index], None, None, self.file_names[index])


class Data():
    def __init__(
        self,
        dataset_dir: str,
        splits: list,
        valid_ratio: float,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        n_mels: int = 128
    ) -> None:
        super().__init__()
        self.train, self.valid, self.test = None, None, None
        self.dataset_dir = dataset_dir
        self.label_dict, self.sent_dict, self.vocab, self.scenario_set = self.load_label()

        # setup transform
        self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, normalized=True)

        for split in splits:
            file_names = os.listdir('%s/%s' % (dataset_dir, split))
            print('%s: %d' % (split, len(file_names)))
            if split == 'train':
                random.shuffle(file_names)
                train_end_idx = int(len(file_names) * (1 - valid_ratio))
                self.train = AudioDataset(
                    mel_specs=self.get_mel_spec(file_names[:train_end_idx], split),
                    scenarios=self.get_scenario(file_names[:train_end_idx]),
                    transcriptions=self.get_transciption(file_names[:train_end_idx]),
                    file_names=file_names[:train_end_idx]
                )
                self.valid = AudioDataset(
                    mel_specs=self.get_mel_spec(file_names[train_end_idx:], split),
                    scenarios=self.get_scenario(file_names[train_end_idx:]),
                    transcriptions=self.get_transciption(file_names[train_end_idx:]),
                    file_names=file_names[train_end_idx:]
                )
            elif split == 'test':
                file_names.sort()
                self.test = AudioDataset(
                    mel_specs=self.get_mel_spec(file_names, split),
                    scenarios=None,
                    transcriptions=None,
                    file_names=file_names
                )
            else:
                raise NotImplementedError
        return

    def load_label(
        self
    ) -> (dict, dict, list):
        label_dict, sent_dict, vocab = {}, {}, set()
        labels = pd.read_csv(
            '%s/train.csv' % self.dataset_dir,
            dtype={'file': str, 'scenario': str, 'sentence': str}
        ).values.tolist()
        scenario_set = ['email', 'play', 'lists', 'qa', 'weather', 'iot', 'audio', 'calendar', 'alarm', 'general']
        print('num_label:', len(scenario_set))
        self.num_label = len(scenario_set)
        for label in labels:
            label_dict.update({label[0]: scenario_set.index(label[1])})
            sent_dict.update({label[0]: label[2]})
            vocab.update(label[2].split(' '))
        vocab = ['<blank>'] + list(vocab)
        return label_dict, sent_dict, vocab, scenario_set

    def sent_to_ids(
        self,
        sentence: str
    ) -> list:
        return [self.vocab.index(token) for token in sentence.split(' ')]

    def get_mel_spec(
        self,
        file_names: list,
        split: str
    ) -> (list, list):
        mel_specs = []
        for file_name in tqdm(file_names):
            waveform, sample_rate = torchaudio.load('%s/%s/%s' % (self.dataset_dir, split, file_name))
            mel_specs.append(self.audio_transform(waveform).log().squeeze(0).transpose(0, 1))
        return mel_specs

    def get_scenario(
        self,
        file_names: list
    ) -> list:
        scenarios = []
        for file_name in file_names:
            scenarios.append(
                torch.tensor(
                    self.label_dict[file_name.split('.')[0]],
                    dtype=torch.long
                )
            )
        return scenarios

    def get_transciption(
        self,
        file_names: list
    ) -> list:
        transcriptions = []
        for file_name in file_names:
            transcriptions.append(
                torch.tensor(
                    self.sent_to_ids(self.sent_dict[file_name.split('.')[0]]),
                    dtype=torch.long
                ).unsqueeze(0).transpose(0, 1)
            )
        return transcriptions


def train_data_processing(
    data: list
) -> None:
    mel_specs = []
    scenarios = []
    transcriptions = []
    input_lengths = []
    label_lengths = []

    for (mel_spec, scenario, transcription, _) in data:
        mel_specs.append(mel_spec)
        scenarios.append(scenario)
        transcriptions.append(transcription)
        input_lengths.append(mel_spec.shape[0] // 2)
        label_lengths.append(len(transcription))

    mel_specs = nn.utils.rnn.pad_sequence(mel_specs, batch_first=True).unsqueeze(1).transpose(2, 3)
    scenarios = torch.stack(scenarios)
    transcriptions = nn.utils.rnn.pad_sequence(transcriptions, batch_first=True).squeeze(2)
    return mel_specs, scenarios, transcriptions, input_lengths, label_lengths


def test_data_processing(
    data: list
) -> None:
    mel_specs = [mel_spec for (mel_spec, _, _, _) in data]
    file_names = [file_name for (_, _, _, file_name) in data]
    mel_specs = nn.utils.rnn.pad_sequence(mel_specs, batch_first=True).unsqueeze(1).transpose(2, 3)
    return mel_specs, file_names


def main():
    return


if __name__ == '__main__':
    main()
