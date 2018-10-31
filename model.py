from torch import nn
from torch.nn.utils import rnn
import torch
import torchvision
import torch.nn.functional as F
import pdb
import numpy as np


# Model that takes packed sequences in training
class UtteranceModel(nn.Module):
    def __init__(self, num_phonema, cnn_compression=1, bidirectional=True):
        super(UtteranceModel, self).__init__()
        self.cnn_compression = cnn_compression
        self.num_phonema = num_phonema
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        # self.lstm_input_size = 320
        self.lstm_input_size = 40
        self.hidden_size = 256
        self.nlayers = 3
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=(1, 2), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ELU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=(1, 1), bias=False),
        #     nn.ELU(),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ELU()
        # )
        self.rnn = nn.LSTM(input_size=self.lstm_input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.nlayers,
                           bidirectional=self.bidirectional)

        # define initial state
        self.init_hidden = torch.nn.Parameter(
                torch.randn(self.nlayers * self.directions, 1, self.hidden_size),
                requires_grad=True)  # num_layers * num_directions, 1, hidden_size
        self.init_cell = torch.nn.Parameter(
                torch.randn(self.nlayers * self.directions, 1, self.hidden_size),
                requires_grad=True)  # num_layers * num_directions, 1, hidden_size

        self.scoring = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.num_phonema)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seq_list):
        '''
        :param seq_list: list of sequences ordered by size
        :return: scores (sum(seq_lens), num_phonema)
        '''
        batch_size = len(seq_list)
        lens = [len(s) for s in seq_list]  # lens of all inputs (sorted by loader)
        # seq_list = rnn.pad_sequence(seq_list, batch_first=True)  # batch_size, max_len, features
        # seq_list = seq_list.cuda() if torch.cuda.is_available() else seq_list
        # seq_list = seq_list.unsqueeze(1)  # create a channel for CNN: batch_size, 1, max_len, features
        # embedding = self.cnn(seq_list)  # wasteful cnn computes over padded data: batch_size, 1, red_max_len, features
        # n, c, h, w = embedding.size()  # h is (CNN reduced) max_len; w * c are features
        # reduced_embeddings = []  # batch_size, |reduced_lens|, features
        # reduced_lens = []
        # for i in range(batch_size):
        #     l = -(-lens[i] // self.cnn_compression)
        #     e = embedding[i, :, :l, :].permute(0, 2, 1).contiguous().view(c * w, l).permute(1, 0)  # seq_len, features
        #     reduced_embeddings.append(e)
        #     reduced_lens.append(l)
        reduced_embeddings = seq_list
        reduced_lens = lens
        packed_input = rnn.pack_sequence(reduced_embeddings)  # packed uneven length sequences for fast RNN processing
        packed_input = packed_input.cuda() if torch.cuda.is_available() else packed_input
        # learn initial state
        self.init_hidden.data = self.init_hidden.data.expand(self.nlayers * self.directions, batch_size, self.hidden_size)
        self.init_cell.data = self.init_cell.data.expand(self.nlayers * self.directions, batch_size, self.hidden_size)
        packed_output, hidden = self.rnn(packed_input, (self.init_hidden, self.init_cell))
        packed_output, _ = rnn.pad_packed_sequence(packed_output)  # unpacked output (padded)
        output_flatten = torch.cat(
            [packed_output[:reduced_lens[i], i] for i in
             range(batch_size)])  # concatenated output (sum(reduced_lens), hidden)
        scores_flatten = self.scoring(
            torch.sigmoid(output_flatten))  # concatenated scores (sum(reduced_lens), num_phonema)
        cum_lens = np.cumsum([0] + reduced_lens)
        scores_unflatten = [scores_flatten[cum_lens[i]:cum_lens[i+1]] for i in range(batch_size)]
        scores_unflatten = rnn.pad_sequence(scores_unflatten)  # max_len, batch, num_phonema
        return scores_unflatten, hidden, reduced_lens  # return concatenated scores, hidden state, length of seqs


class AudioDenseNet(nn.Module):
    '''
    Implementation heavily inspired by Torchvision's Densenet.
    This is the model used for final output.
    '''

    def __init__(self, classnum):
        super(AudioDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=(2, 1), padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
            # blocks of (6, 12, 24, 16), 64 initial features
            torchvision.models.densenet._DenseBlock(num_layers=6, num_input_features=16,
                                                    bn_size=4, growth_rate=16, drop_rate=0),
            torchvision.models.densenet._Transition(num_input_features=112, num_output_features=112 // 2),
            torchvision.models.densenet._DenseBlock(num_layers=12, num_input_features=56,
                                                    bn_size=4, growth_rate=16, drop_rate=0),
            torchvision.models.densenet._Transition(num_input_features=248, num_output_features=248 // 2),
            torchvision.models.densenet._DenseBlock(num_layers=24, num_input_features=124,
                                                    bn_size=4, growth_rate=16, drop_rate=0),
            # torchvision.models.densenet._Transition(num_input_features=520, num_output_features=250 // 2),
            # torchvision.models.densenet._DenseBlock(num_layers=16, num_input_features=125,
            #                                         bn_size=4, growth_rate=16, drop_rate=0),
            nn.BatchNorm2d(508)
        )

        # self.strider = nn.Conv2d(1, 3, (7, 15), (1, 8), (3, 0))
        # self.avg_pool = nn.AvgPool2d(kernel_size=(1, 29))
        # self.embeddings = nn.Linear(4096, 300, bias=False)
        # self.clf = nn.Linear(300, classnum, bias=False)
        # if torch.cuda.is_available():
        #     self.alpha = torch.from_numpy(np.array(16)).float().cuda()
        # else:
        #     self.alpha = torch.from_numpy(np.array(16)).float()

        # initialize
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.strider(x)
        # x = F.elu(x)
        x = self.features(x)
        x = F.relu(x, inplace=True)
        # x = self.avg_pool(x).view(x.size(0), -1)
        # x = self.embeddings(x)
        # x = F.normalize(x) * self.alpha
        # x = F.sigmoid(x)
        # x = self.clf(x)
        return x


class DigitsModel(nn.Module):
    def __init__(self, num):
        super(DigitsModel, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(1, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=(1, 1), bias=False),
            nn.ELU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

    def forward(self, x):
        x = self.embed(x)
        return x


if __name__ == '__main__':
    import phoneme_list
    import torchsummary

    model = UtteranceModel(len(phoneme_list.PHONEME_MAP) + 1)
    with torch.no_grad():
        # torchsummary.summary(model, (1, 2700, 40))
        # max 2700, 630 average,
        # print(model(torch.ones((20, 1, 5, 40))).shape)
        print(model([torch.ones((100, 40)), torch.ones((90, 40))])[0].shape)
