import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from UtilisSet.optimization_layout.Layers import FullyConnectedNN
import torch
import numpy as np

class EGATE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_clusters,heads=3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads,fill_value=1)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=heads,fill_value=1)
        self.n_clusters = n_clusters
        self.linear_p = nn.Linear(out_channels* heads,self.n_clusters)
    def encoder(self, data):
        x, edge_index,edge_attr=data.x,data.edge_index,data.edge_attr
        num_x = torch.nan_to_num(x, nan=0.0)
        num_x = F.elu(self.gat1(num_x, edge_index,edge_attr))
        num_x = self.gat2(num_x, edge_index, edge_attr)
        return F.normalize(num_x,p=2,dim=1)

    def forward(self, data):
        z = self.encoder(data)
        A_hat = torch.matmul(z,z.t())
        p_logits = self.linear_p(z)
        p = F.softmax(p_logits)

        return A_hat,p,z
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads,fill_value=1)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=heads,fill_value=1)

    def forward(self, data):
        x, edge_index,edge_attr=data.x,data.edge_index,data.edge_attr
        num_x = torch.nan_to_num(x, nan=0.0)
        num_x = F.elu(self.gat1(num_x, edge_index,edge_attr))
        num_x = self.gat2(num_x, edge_index, edge_attr)
        return num_x

class MLPDecoder1(nn.Module):
    def __init__(self, embedding_dim,heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim*2*heads, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z, edge_index):
        src, dst = edge_index
        edge_feat = torch.cat([z[src], z[dst]], dim=1)
        return self.mlp(edge_feat).squeeze()

class MLPDecoder2(nn.Module):
    def __init__(self, embedding_dim,heads,num_clusters):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim*heads, 64),
            nn.ReLU(),
            nn.Linear(64, num_clusters)
        )

    def forward(self, z):
        return F.sigmoid(self.mlp(z))
class MLPGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_clusters,heads=3):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, out_channels, heads)
        self.decoder1 = MLPDecoder1(out_channels,heads)
        self.decoder2 = MLPDecoder2(out_channels, heads,n_clusters)

    def forward(self,data):
        x,edge_index,neg_edge_index = data.x,data.edge_index,data.neg_edge_index
        z = self.encoder(data)

        pos_pred = self.decoder1(z, edge_index)
        neg_pred = self.decoder1(z, neg_edge_index)

        logits_pred = self.decoder2(z)

        pos_label = torch.ones(pos_pred.size(0), device=z.device)
        neg_label = torch.zeros(neg_pred.size(0), device=z.device)

        all_link_pred = torch.cat([pos_pred, neg_pred], dim=0)
        all_link_label = torch.cat([pos_label, neg_label], dim=0)

        return all_link_pred,all_link_label,logits_pred,z



class Transformer_GINEConv_NN(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, n_clusters,heads=3
    ):
        super(Transformer_GINEConv_NN, self).__init__()
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.output_dim = out_channels

        self.skip_alpha = nn.Parameter(torch.tensor(0.8))

        self.non_linearity = "PReLU"
        self.n_hidden_layers = 1
        self.n_gcn_layers = 11
        self.eps_gnn = 0.5
        self.number_edge_inputs = 4
        self.linear_p = nn.Linear(hidden_channels, n_clusters)
        self.create_layers_dict()
    def create_layers_dict(self):
        self._nodeEncoder = FullyConnectedNN(
            self.input_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
        )
        self._edgeEncoder = FullyConnectedNN(
            self.number_edge_inputs,
            self.hidden_dim,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
        )

        _mlp_for_gineconv = FullyConnectedNN(
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
        )

        self._processor = GINEConv(
            _mlp_for_gineconv, eps=self.eps_gnn, train_eps=True
        )  # .jittable()

        self._nodeDecoder = FullyConnectedNN(
            self.hidden_dim,
            self.output_dim,
            self.hidden_dim,
            self.n_hidden_layers,
            with_bias=True,
            non_linearity=self.non_linearity,
            final_bias=False,
        )

        self.layers_dict = nn.ModuleDict(
            {
                "nodeEncoder": self._nodeEncoder,
                "edgeEncoder": self._edgeEncoder,
                "processor": self._processor,
                "nodeDecoder": self._nodeDecoder,
            }
        )

    def forward(self, win):
        edge_features = win.edge_attr
        node_features = win.x
        coded_x = self.layers_dict["nodeEncoder"](node_features)
        coded_e_i = self.layers_dict["edgeEncoder"](edge_features)

        for s in range(self.n_gcn_layers):
            spatio_process_x = self.layers_dict["processor"](coded_x, win.edge_index, coded_e_i)

        # z = self.layers_dict["nodeDecoder"](spatio_process_x)
        z = spatio_process_x
        A_hat = torch.matmul(z,z.t())
        p_logits = self.linear_p(z)
        p = F.softmax(p_logits)

        return A_hat,p,z

