import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.data import Data
class FeatureConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels,num_kernels=10):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels,out_channels,kernel_size=2*k+1,padding=k),
                                                 nn.BatchNorm1d(out_channels),nn.ReLU()) for k in range(num_kernels)])
        self.fc = nn.Sequential(
            nn.Linear(out_channels * num_kernels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, out_channels)
        )
        self.final_bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        feats = []
        for conv in self.convs:
            h = conv(x)               # → (B·N, out_channels, seq_len)
            feats.append(h)
        h = torch.cat(feats, dim=1)   # → (B·N, out_channels * num_kernels)
        h = h.permute(0, 2, 1)
        h = self.fc(h)
        h = h.permute(0, 2, 1)
        h = self.final_bn(h)          # → (B·N, out_channels)
        return h
class PhysicalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        alpha = self.attention_layer(h)
        return alpha
class Conv_GINE_NN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim,edge_index,edge_attr, hidden_dim, num_layers, total_node_num):
        super().__init__()

        self.total_node_num = total_node_num
        self.hidden_dim = hidden_dim
        # 注册图结构为buffer（非训练参数）
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        # 1. 时序特征提取部分（堆叠多层）
        self.temporal_encoder1 = FeatureConvLayer(in_channels=1,out_channels=hidden_dim)
        self.temporal_encoder2 = FeatureConvLayer(in_channels=1, out_channels=hidden_dim)
        self.edge_encoder2 = FeatureConvLayer(in_channels=edge_input_dim, out_channels=hidden_dim)
        # 2. 未监测节点可学习特征（初始化为可训练参数）
        self.unmonitored_feature = nn.Parameter(torch.randn(total_node_num, 24,hidden_dim))

        # 3. 多层GINE卷积（含残差）
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # self.gnn_layers.append(GINEConv(mlp, edge_dim=edge_input_dim))
            self.gnn_layers.append(GINEConv(mlp,edge_dim=edge_input_dim))
        self.phy_attention = PhysicalAttention(hidden_dim)
        # 4. 输出液位预测
        self.final_linear = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)
        )
        self.gru = nn.GRU(hidden_dim,hidden_dim,num_layers=2,batch_first=True)
        self.trans_layer = nn.Linear(hidden_dim*2,hidden_dim)
        # self.final_linear = nn.Linear(hidden_dim, 1)
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data):
        """
        x_monitor: (B, M, 1, T) - 观测节点液位数据
        edge_index: (2, E)
        edge_attr: (E, edge_input_dim)
        """

        B = data.batch_size
        T = data.norm_runoff.size(-1)
        runoff = data.norm_runoff.view(B,-1,T)
        N = runoff.size(1)

        monitored_num = len(data.input_mask)
        monitored_head = data.norm_h_x.view(B,N,T)[:,data.input_mask,:]
        x_monitored = monitored_head.unsqueeze(1)
        x_monitored = x_monitored.permute(0, 2, 1, 3)
        x_monitored = x_monitored.reshape(B * monitored_num, 1, T)
        x_monitored_dim = self.temporal_encoder1(x_monitored)
        x_monitored_dim = x_monitored_dim.view(B,monitored_num,self.hidden_dim,T)
        x_monitored_dim = x_monitored_dim.permute(0,1,3,2)
        x_unmonitored_dim = self.unmonitored_feature.unsqueeze(0).expand(B, -1, -1, -1)

        batch_idx = torch.arange(B, device=runoff.device).unsqueeze(-1)  # (B, 1)
        monitor_idx_tensor = data.input_mask.unsqueeze(0).expand(B, -1)  # (B, M)

        xt = x_unmonitored_dim.clone()
        xt[batch_idx, monitor_idx_tensor] = x_monitored_dim
        xt = xt.permute(0,2,1,3)
        xt = xt.reshape(B*T,N,-1)

        x_runoff = runoff.unsqueeze(1)
        x_runoff = x_runoff.permute(0, 2, 1, 3)
        x_runoff = x_runoff.reshape(B * N, 1, T)
        x_runoff_dim = self.temporal_encoder2(x_runoff)
        x_runoff_dim = x_runoff_dim.view(B,N,self.hidden_dim,T)
        x_runoff_dim = x_runoff_dim.permute(0,3,1,2)
        xr = x_runoff_dim.reshape(B * T, N, -1)
        ####第一层
        x_gcn = []
        for b_t in xt:
            h = b_t
            for gnn in self.gnn_layers:
                h_new = gnn(h, self.edge_index, self.edge_attr)
                h = self.phy_attention(h)*h+(1-self.phy_attention(h))*h_new
            x_gcn.append(h)
        x_gcn = torch.stack(x_gcn,dim=0)#[B*T,N,F]
        x_gcn = x_gcn.view(B,T,N,-1)
        x_gcn = x_gcn.permute(2,0,1,3)

        out_sequence = []
        for node_seq in x_gcn:
            out, _ = self.gru(node_seq)
            out_sequence.append(out)
        # 合并所有节点
        out_sequence = torch.stack(out_sequence, dim=1)  # (B, N, hidden_dim, T)
        ##第二层
        reshape_out = out_sequence.permute(0,2,1,3)
        reshape_out = reshape_out.reshape(B*T,N,-1)
        combine_xr = torch.cat([reshape_out,xr],dim=-1)
        second_xt = self.trans_layer(combine_xr)

        x_gcn = []
        for b_t in second_xt:
            h = b_t
            for gnn in self.gnn_layers:
                h_new = gnn(h, self.edge_index, self.edge_attr)
                h = self.phy_attention(h)*h+(1-self.phy_attention(h))*h_new
            x_gcn.append(h)
        x_gcn = torch.stack(x_gcn,dim=0)#[B*T,N,F]
        x_gcn = x_gcn.view(B,T,N,-1)
        x_gcn = x_gcn.permute(2,0,1,3)

        out_sequence = []
        for node_seq in x_gcn:
            out, _ = self.gru(node_seq)
            out_sequence.append(out[:,-12:])
        out_sequence = torch.stack(out_sequence, dim=1)
        # 5. 输出液位预测
        outputs = self.final_linear(out_sequence).squeeze(-1)  # (B, N, T, 1)
        outputs = outputs.reshape(B*N,-1)

        return outputs
