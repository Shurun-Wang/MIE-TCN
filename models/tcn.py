#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as nn

class TCNet(nn.Module):
    """
    main structure
    """
    def __init__(self, emb_size, num_channels, kernel_size, dropout):
        super(TCNet, self).__init__()
        self.tcanet = TemporalConvNet(emb_size, num_channels, kernel_size, dropout)
        self.gelu = nn.ELU()

    def forward(self, input):
        input = input.permute(0, 2, 1)
        out = self.tcanet(input)  # input dimension (batch, channel, seq_length)
        out = self.gelu(out[:, :, -1])
        return out


class TemporalConvNet(nn.Module):
    """
    second main structure
    """
    def __init__(self, emb_size, num_channels, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = emb_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batchsize, seq_len, emb_size]
        return self.network(x)


class TemporalBlock(nn.Module):
    """
    Block structure
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # self.attention = AttentionBlock(n_inputs, key_size, n_inputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.net = self._make_layers(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout)
        self.elu = nn.ELU()

    def _make_layers(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        layers_list = []
        # n_inputs = 1, n_outputs = 25, kernel_size = 7, stride = 1
        layers_list.append(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        layers_list.append(Chomp1d(padding))
        layers_list.append(nn.BatchNorm1d(n_outputs))
        layers_list.append(nn.ELU())
        layers_list.append(nn.Dropout(dropout))
        layers_list.append(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        layers_list.append(Chomp1d(padding))
        layers_list.append(nn.BatchNorm1d(n_outputs))
        layers_list.append(nn.ELU())
        layers_list.append(nn.Dropout(dropout))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        # x: [N, emb_size, T]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)


class Chomp1d(nn.Module):
    """
    just a tool for cutting the extra padding
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


if __name__ == '__main__':
    pass