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

