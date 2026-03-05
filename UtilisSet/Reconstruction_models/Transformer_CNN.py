import math
import torch
import torch.nn as nn



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #print('##posi',position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #print('##div_term',div_term.shape)
        #print((position * div_term).shape)
        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x=x.transpose(0,1)
        return x + self.pe[:x.size(0), :]
class FusedCNNTransformer(nn.Module):
    def __init__(
        self, embed_dim=64,cnn_channels=64,kernel_size=3,
            num_transformer_layers=2,nhead=4,dropout=0.1,fc_dim=32
    ):
        super(FusedCNNTransformer, self).__init__()
        self.embed_dim=embed_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim,out_channels=cnn_channels,kernel_size=kernel_size,
                     padding=kernel_size//2),nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels,out_channels=cnn_channels,kernel_size=kernel_size,
                      padding=kernel_size//2),nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=nhead,dropout=dropout)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layers,num_layers=num_transformer_layers)
        self.gate_fc = nn.Linear(cnn_channels+embed_dim,cnn_channels+embed_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,x):
        batch_size,seq_len,_ = x.size()
        x_pos = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x_pos)
        transformer_feature = transformer_output.transpose(0,1)
        cnn_input = x.transpose(1,2)
        cnn_feature = self.cnn(cnn_input)
        cnn_feature = cnn_feature.transpose(2,1)
        fused = torch.cat([cnn_feature,transformer_feature],dim=-1)
        gate = torch.sigmoid(self.gate_fc(fused))
        fused_features = gate*fused
        return fused_features