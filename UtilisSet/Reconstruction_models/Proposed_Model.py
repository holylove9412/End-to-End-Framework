import copy
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import scatter
from torch_geometric.nn import GINEConv
from UtilisSet.Reconstruction_models.Transformer_CNN import FusedCNNTransformer
from UtilisSet.Reconstruction_models.Layers import FullyConnectedNN


class BaseModel(nn.Module):

    def forward(self, win):
        win = copy.deepcopy(win)
        is_batch = win.batch is not None

        h0 = win.x
        ground_level = win.node_attr[:,-2:-1]
        win['norm_elevation'] = win.node_attr[:,-1:]

        transformed_x,nan_node_mask,nan_edge_mask = self._transform_x_with_layers(win, h0)
        pred_head = self._add_skip_connection(win, h0, transformed_x)

        pred_head_clipped = torch.min(pred_head, ground_level)
        pred_head_lower_clipped = torch.max(pred_head_clipped, win.norm_elevation)

        return pred_head_lower_clipped
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _transform_x_with_layers(self, win, h0):
        if self.aggregation_type == "Combined":
            transformed_x = self._use_layers_in_forward_pass_combined(win, h0)
        elif self.aggregation_type == "Separated":
            transformed_x,nan_node_mask,nan_edge_mask = self._use_layers_in_forward_pass_separated(win, h0)
        else:
            raise Exception("Unknown aggregation")
        return transformed_x,nan_node_mask,nan_edge_mask

    def _add_skip_connection(self, win, h0, transformed_x):
        elevation = win.norm_elevation.expand(-1,transformed_x.size(-1))
        prev_y = h0 - elevation
        pred_y_skipped = (
            self.skip_alpha * transformed_x + (1.0 - self.skip_alpha) * prev_y
        )
        pred_head = pred_y_skipped + elevation
        return pred_head


    def _clean_nan_and_get_mask(self,tensor, fill_value=0.0):
        nan_mask = torch.isnan(tensor)
        cleaned_tensor = torch.nan_to_num(tensor, nan=fill_value)
        return cleaned_tensor, nan_mask

    def _use_layers_in_forward_pass_separated(self, win, h0):
        edge_features = self._get_edge_features(win)
        # node_features = self._get_one_step_features_node(win, h0)
        node_features = self._get_node_features(win)
        T =node_features.size(1)
        clean_node_features,nan_node_mask = self._clean_nan_and_get_mask(node_features)
        clean_edge_features, nan_edge_mask = self._clean_nan_and_get_mask(edge_features)
        coded_x = self.layers_dict["nodeEncoder"](clean_node_features)
        coded_e_i = self.layers_dict["edgeEncoder"](clean_edge_features)

        spatio_out = []
        for t in range(T):
            spatio_process_x= coded_x[:,t,:].clone()
            for s in range(self.n_gcn_layers):
                spatio_process_x = self.layers_dict["processor"](spatio_process_x, win.edge_index, coded_e_i)
            spatio_out.append(spatio_process_x)
        spatio_out = torch.stack(spatio_out,dim=1)        #
        # time_out,_ = self.gru(spatio_out)
        time_out = self.transformer_cnn(spatio_out)
        decoded_x = self.layers_dict["nodeDecoder"](time_out)
        return decoded_x.squeeze(-1),nan_node_mask,nan_edge_mask

    def _use_layers_in_forward_pass_combined(self, win, h0):
        edge_features = self._get_edge_features(win)
        node_features = self._get_one_step_features_node(win, h0)

        source, target = win.edge_index

        node_features_in_source = node_features[source]
        node_features_in_target = node_features[target]

        mixed_features = torch.cat(
            [edge_features, node_features_in_source, node_features_in_target], axis=1
        )

        coded_e_i = self.layers_dict["edgeEncoderMix"](mixed_features)
        coded_x = scatter(coded_e_i, source, dim=0, reduce="sum")

        processed_x = self.layers_dict["processor"](coded_x, win.edge_index)

        decoded_x = self.layers_dict["nodeDecoder"](processed_x)
        return decoded_x
    def _data_mask(self,data):
        monitored_nodes=torch.tensor([54,187,104,202,335,298])
        train_monitores_nodes = monitored_nodes[torch.randperm(len(monitored_nodes))[:4]]
        mask = torch.full_like(data,fill_value=0).reshape(-1,362,data.shape[-1])

        mask[:,train_monitores_nodes,:] = 1

        return mask.reshape(-1,data.shape[-1])
    def _get_one_step_features_node(self, win, h0):
        runoff_step = win.norm_runoff
        # runoff_step = win.norm_runoff[
        #               :, step: step + self.steps_behind + self.prediction_steps
        #               ]
        # mask = self._data_mask(h0)
        one_step_x = torch.cat((h0, win.norm_elevation), dim=1)##节点底高程、水头、径流

        return one_step_x
    def _get_node_features(self,win):
        # node_attributes=win.node_attr
        h_x = win.node_attr[:,:24]
        delta_h = win.node_attr[:,24:48]
        slope_h = win.node_attr[:,48:72]
        ground_h = win.node_attr[:,-2:-1].expand(-1,h_x.size(-1))
        elevation_h = win.node_attr[:,-1:].expand(-1,h_x.size(-1))
        node_features=torch.cat([h_x.unsqueeze(-1),delta_h.unsqueeze(-1),
                       slope_h.unsqueeze(-1),ground_h.unsqueeze(-1),
                       elevation_h.unsqueeze(-1)],dim=-1)

        return node_features

    def _get_edge_features(self, win):

        edge_attributes = win.edge_attr
        return edge_attributes

    def _assert_valid_length(self, length_simulation):
        assert (
            length_simulation >= self.prediction_steps
        ), "The prediction is longer than the desired simulation length."
        assert (
            length_simulation % self.prediction_steps == 0
        ), "The prediction should be a multiple of the simulation length."

    def _get_new_h0(self, old_h0, new_h):
        original_size = old_h0.shape[1]
        concatenated = torch.cat((old_h0, new_h), dim=1)
        new_h0 = concatenated[:, -original_size:]
        return new_h0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class Edge_STGCN(BaseModel):
    def __init__(
        self, steps_behind, hidden_dim, skip_alpha, prediction_steps=1, **kwargs
    ):
        super(Edge_STGCN, self).__init__()
        self.aggregation_type = "Separated"

        self.steps_behind = steps_behind
        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps

        self.skip_alpha = nn.Parameter(torch.tensor(skip_alpha))
        self.length_window = 3 * self.steps_behind
          # Steps behind (depth), steps behind (runoff), prediction_steps (runoff)
        self.non_linearity = kwargs["non_linearity"]
        self.n_hidden_layers = kwargs["n_hidden_layers"]
        self.n_gcn_layers = kwargs["n_gcn_layers"]
        self.eps_gnn = kwargs["eps_gnn"]
        self.edge_input_list = kwargs["edge_input_list"]
        self.number_edge_inputs = len(self.edge_input_list)
        self.transformer_cnn = FusedCNNTransformer()
        self.gru = nn.GRU(hidden_dim,hidden_dim,2,batch_first=True)

        self.create_layers_dict()

    def create_layers_dict(self):
        self._nodeEncoder = FullyConnectedNN(
            5,
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
            2*self.hidden_dim,
            1,
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
