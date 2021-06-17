import torch.nn as nn
import torch.nn.functional as F


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    def __init__(self, vocab_size, num_label, rnn_dim=512, n_feats=128, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2),
            nn.Conv2d(32, 32, 3, stride=1, padding=3 // 2),
            nn.Conv2d(32, 32, 3, stride=1, padding=3 // 2)
        )
        self.fc_1 = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(
                rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                hidden_size=rnn_dim,
                dropout=dropout,
                batch_first=(i == 0)
            ) for i in range(5)
        ])
        self.fc_2 = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),                   # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier_transciption = nn.Linear(rnn_dim, vocab_size)
        self.classifier_scenario = nn.Linear(rnn_dim, num_label)
        return

    def forward(self, x):
        x = self.cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])     # (batch, feature, time)
        x = x.transpose(1, 2)                                   # (batch, time, feature)
        x = self.fc_1(x)
        x = self.birnn_layers(x)
        x = self.fc_2(x)
        t = self.classifier_transciption(x)
        s = self.classifier_scenario(x[:, -1, :])
        return s, t


def main():
    return


if __name__ == '__main__':
    main()
