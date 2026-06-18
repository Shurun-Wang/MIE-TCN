import torch
import torch.nn as nn
from models.encoder import Encoder, AttentionBlock, DistillBlock
from models.attn import ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.tcn import TCNet


class MIE_TCN(nn.Module):
    def __init__(self, enc_in, factor=5, d_model=512, n_heads=8, e_layers=2, d_ff=512,
                 dropout=0.0, activation='ELU', output_attention=False, length=48, output_size=6):
        super(MIE_TCN, self).__init__()

        self.output_attention = output_attention
        self.output_size = output_size
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, length)
        # Encoder
        att_list, distill_list = [], []
        for l in range(e_layers):
            att = ProbAttention(factor, attention_dropout=dropout, output_attention=output_attention)
            att_layer = AttentionLayer(probattention=att, d_model=d_model, n_heads=n_heads)
            att_list.append(AttentionBlock(att_layer, d_model, d_ff, dropout=dropout, activation=activation))
        for l in range(e_layers - 1):
            distill_list.append(DistillBlock(d_model))
        self.encoder = Encoder(attn_layers=att_list, distill_layers=distill_list,
                               norm_layer=torch.nn.LayerNorm(d_model))
        # TCN
        self.tcn = TCNet(emb_size=d_model, num_channels=[256, 128, 64], kernel_size=3, dropout=dropout)
        # 自适应输出维度
        self.projection = nn.Linear(64, output_size)

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        tcn_out = self.tcn(enc_out)
        tcn_out = self.projection(tcn_out)

        if self.output_attention:
            return tcn_out, attns
        else:
            return tcn_out


class MIE_LSTM(nn.Module):
    def __init__(self, enc_in, factor=5, d_model=512, n_heads=8, e_layers=2, d_ff=512,
                 dropout=0.0, activation='ELU', output_attention=False, length=48, output_size=6):
        super(MIE_LSTM, self).__init__()

        self.output_attention = output_attention
        self.output_size = output_size
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, length)
        # Encoder
        att_list, distill_list = [], []
        for l in range(e_layers):
            att = ProbAttention(factor, attention_dropout=dropout, output_attention=output_attention)
            att_layer = AttentionLayer(probattention=att, d_model=d_model, n_heads=n_heads)
            att_list.append(AttentionBlock(att_layer, d_model, d_ff, dropout=dropout, activation=activation))
        for l in range(e_layers - 1):
            distill_list.append(DistillBlock(d_model))
        self.encoder = Encoder(attn_layers=att_list, distill_layers=distill_list,
                               norm_layer=torch.nn.LayerNorm(d_model))
        # LSTM
        self.num_layers = 1
        self.hidden_size = int(length / 2)
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=self.hidden_size,
                             num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                             num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                             num_layers=self.num_layers, batch_first=True, dropout=0.3)
        # 自适应输出维度：动态创建 output_size 个 fc 层
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(output_size)])

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        x, attns = self.encoder(enc_out)

        batch_size, seq_len = x.shape[0], x.shape[1]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))

        h1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm2(out, (h1, c1))

        h2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm3(out, (h2, c2))

        # 动态生成 output_size 个预测
        preds = [fc(out)[:, -1, :] for fc in self.fcs]
        pred = torch.cat(preds, dim=1)
        return pred


class MIE_GRU(nn.Module):
    def __init__(self, enc_in, factor=5, d_model=512, n_heads=8, e_layers=2, d_ff=512,
                 dropout=0.0, activation='ELU', output_attention=False, length=48, output_size=6):
        super(MIE_GRU, self).__init__()

        self.output_attention = output_attention
        self.output_size = output_size
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, length)
        # Encoder
        att_list, distill_list = [], []
        for l in range(e_layers):
            att = ProbAttention(factor, attention_dropout=dropout, output_attention=output_attention)
            att_layer = AttentionLayer(probattention=att, d_model=d_model, n_heads=n_heads)
            att_list.append(AttentionBlock(att_layer, d_model, d_ff, dropout=dropout, activation=activation))
        for l in range(e_layers - 1):
            distill_list.append(DistillBlock(d_model))
        self.encoder = Encoder(attn_layers=att_list, distill_layers=distill_list,
                               norm_layer=torch.nn.LayerNorm(d_model))
        # GRU
        self.num_layers = 1
        self.hidden_size = 300
        self.gru1 = nn.GRU(input_size=512, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, batch_first=True, dropout=0.1)
        self.gru2 = nn.GRU(input_size=300, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, batch_first=True, dropout=0.1)
        # 自适应输出维度：动态创建 output_size 个 fc 层
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(output_size)])

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        x, attns = self.encoder(enc_out)

        out, _ = self.gru1(x)
        out, _ = self.gru2(out)

        # 动态生成 output_size 个预测
        preds = [fc(out)[:, -1, :] for fc in self.fcs]
        pred = torch.cat(preds, dim=1)
        return pred


class MIE_MLP(nn.Module):
    def __init__(self, enc_in, factor=5, d_model=512, n_heads=8, e_layers=2, d_ff=512,
                 dropout=0.0, activation='ELU', output_attention=False, length=48, output_size=6):
        super(MIE_MLP, self).__init__()

        self.output_attention = output_attention
        self.output_size = output_size
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, length)
        # Encoder
        att_list, distill_list = [], []
        for l in range(e_layers):
            att = ProbAttention(factor, attention_dropout=dropout, output_attention=output_attention)
            att_layer = AttentionLayer(probattention=att, d_model=d_model, n_heads=n_heads)
            att_list.append(AttentionBlock(att_layer, d_model, d_ff, dropout=dropout, activation=activation))
        for l in range(e_layers - 1):
            distill_list.append(DistillBlock(d_model))
        self.encoder = Encoder(attn_layers=att_list, distill_layers=distill_list,
                               norm_layer=torch.nn.LayerNorm(d_model))
        # MLP
        in_length = int(length / 2)
        self.fc1 = nn.Linear(in_length * 512, 25)
        self.fc2 = nn.Linear(25, 100)
        self.fc3 = nn.Linear(100, 100)
        # 自适应输出维度
        self.fc4 = nn.Linear(100, output_size)
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.3)

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        x, attns = self.encoder(enc_out)
        x = x.view(x.shape[0], -1)
        out = self.fc1(x)
        out = self.elu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.elu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.elu3(out)
        out = self.drop3(out)
        out = self.fc4(out)
        return out