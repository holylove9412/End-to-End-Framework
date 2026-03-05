import torch.nn as nn
import torch
from torch_geometric.nn import NNConv


class WaterLevelModel(nn.Module):
    def __init__(self, edge_index, edge_attr, input_dim=1, hidden_dim=16):
        super(WaterLevelModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

        edge_feat_dim = edge_attr.shape[1]

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim * hidden_dim)
        )

        self.conv = NNConv(hidden_dim, hidden_dim, self.edge_mlp, aggr='mean', root_weight=True)

        self.out_linear = nn.Linear(hidden_dim, 1)
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, data):

        runoff = data.norm_runoff
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
        num_nodes = len(data.name_nodes[0])
        B_N, T= runoff.shape
        B = B_N // num_nodes
        x_runoff = runoff.view(B, num_nodes, T).transpose(1, 2)
        x_runoff = x_runoff.unsqueeze(-1)
        x_runoff = x_runoff.reshape(T, B * num_nodes, -1)

        hidden = torch.zeros(B * num_nodes, self.hidden_dim, device=x_runoff.device)
        preds = []
        for t in range(T):
            x_t = x_runoff[t]
            hidden_candidate = self.gru_cell(x_t, hidden)

            hidden = self.conv(hidden_candidate, self.edge_index, self.edge_attr)
            out_t = self.out_linear(hidden)
            out_t = out_t.squeeze(1)
            preds.append(out_t)

        pred_levels = torch.stack(preds, dim=0)
        return pred_levels
