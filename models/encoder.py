import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillBlock(nn.Module):
    def __init__(self, c_in):
        super(DistillBlock, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, attention, d_model, d_ff=2048, dropout=0.1, activation="ELU"):
        super(AttentionBlock, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.elu

    def forward(self, x):

        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.activation(self.conv2(y).transpose(-1,1)))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, distill_layers, norm_layer):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.distill_layers = nn.ModuleList(distill_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        attns = []

        for attn_layer, distill_layer in zip(self.attn_layers, self.distill_layers):
            x, attn = attn_layer(x)
            x = distill_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x)
        attns.append(attn)

        x = self.norm(x)

        return x, attns

