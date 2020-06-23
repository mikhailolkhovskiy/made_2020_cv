import numpy as np
import torch
from torch import nn
from torchvision import models
from .common import abc


class RecognitionModel(nn.Module):
    def __init__(
        self,
        cnn_input_size=(64, 320),
        cnn_output_len=20,
        rnn_hidden_size=256,
        rnn_num_layers=2,
        rnn_dropout=0.3,
        num_directions=1
    ):
        # see workshop #9 for more details
        # (https://github.com/BorisLestsov/MADE/blob/master/seminar9-carocr/9%20-%20crnn%20-%20completed.ipynb)
        super(RecognitionModel, self).__init__()
        self.abc = abc
        self.num_classes = len(self.abc)
        self.num_directions = num_directions

        h, w = cnn_input_size
        resnet = getattr(models, 'resnet34')(pretrained=True) # TODO: is it most optimal choice ?
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = nn.Conv2d(w // 32, cnn_output_len, kernel_size=1)

        cnn_num_output_features = self.cnn[-1][-1].bn2.num_features
        # TODO: move these parameters to the constructor function in order experiment using factories & configs
        # TODO: try other RNNs or even CNNs, if you've got enough time
        print('cnn_num_output_features', cnn_num_output_features)
        self.rnn = nn.GRU(input_size=cnn_num_output_features,
                          hidden_size=rnn_hidden_size,
                          num_layers=rnn_num_layers,
                          batch_first=False,
                          dropout=rnn_dropout,
                          bidirectional=(num_directions == 2))

        self.linear = nn.Linear(rnn_hidden_size * num_directions, len(abc)+1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.rnn.num_layers * self.num_directions,
                             batch_size,
                             self.rnn.hidden_size)
        hidden = hidden.to(device)
        return hidden

    def forward(self, x, decode=False):
        hidden = self.init_hidden(x.size(0), next(self.parameters()).device)
        features = self.cnn(x)
        features = self.pool(features)
        sequence = self.features_to_sequence(features)
        sequence, hidden = self.rnn(sequence, hidden)
        sequence = self.linear(sequence)

        if not self.training:
            sequence = self.softmax(sequence)
            if decode:
                sequence = self.decode(sequence)

        return sequence

    def features_to_sequence(self, features):
        b, c, h, w = features.size()
        if h != 1:
            print(features.size())
        assert h == 1, '1 != {}'.format(h)
        features = features.permute(0, 3, 2, 1).contiguous()
        features = self.proj(features)
        features = features.permute(1, 0, 2, 3).contiguous()
        features = features.squeeze(2)
        return features

    def get_block_size(self, layer):
        return layer[-1][-1].bn2.weight.size()[0]

    def pred_to_string(self, pred):
        seq = []
        for i in range(len(pred)):
            label = np.argmax(pred[i])
            seq.append(label - 1)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != -1:
                    out.append(seq[i])
            else:
                if seq[i] != -1 and seq[i] != seq[i - 1]:
                    out.append(seq[i])
        out = ''.join([self.abc[c] for c in out])
        return out

    def decode(self, pred):
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        outputs = []
        for i in range(len(pred)):
            outputs.append(self.pred_to_string(pred[i]))
        return outputs
