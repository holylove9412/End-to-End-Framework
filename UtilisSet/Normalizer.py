
import numpy as np
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod
from torch_geometric.data import Data

def NormalizerFactory(normalizer_name):
    available_normalizer = {
        "Normalizer": Normalizer,
    }
    normalizer = available_normalizer[normalizer_name]
    return normalizer


class abstractNormalizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def normalize_window(self, window):
        pass


class Normalizer(abstractNormalizer):

    def __init__(self, training_windows, abs_flows=False):
        self.training_windows = training_windows
        self.abs_flows = abs_flows
        self.name_nodes = self.training_windows[0].name_nodes
        self.name_conduits = self.training_windows[0].name_conduits

        self.depths_normalizer = self._InternalNormalizer(
            self, ["h_x", "h_y", "elevation", "ground_level"]
        )
        # self.flows_normalizer = self._InternalNormalizer(self, ["q_x", "q_y"])
        self.deltah_normalizer = self._GlobalZScoreNormalizer(self, ["delta_h"])
        self.slopeh_normalizer = self._GlobalZScoreNormalizer(self, ["slope_h"])

        self.length_normalizer = self._InternalNormalizer(self, ["length"])
        self.geom_1_normalizer = self._InternalNormalizer(self, ["height"])
        self.runoff_normalizer = self._InternalNormalizer(self, ["runoff"])
        self.node_slope_normalizer = self._InternalNormalizer(self, ["node_slope"])
        self.conduit_slope_normalizer = self._InternalNormalizer(self, ["conduit_slope"])
        self.volume_normalizer = self._InternalNormalizer(
            self, ["aprox_conduit_volume"]
        )
        self.training_windows = None

    def extract_reduced_data(slef,original_data, keys):
        new_data_dict = {}
        for k in keys:
            v = getattr(original_data, k)
            if isinstance(v, torch.Tensor):
                new_data_dict[k] = v.clone()
            else:
                new_data_dict[k] = v
        return Data(**new_data_dict)

    def prune_data(self,data: Data, keep_keys: list) -> Data:
        current_keys = list(data.keys())
        for key in current_keys:
            if key not in keep_keys:
                del data[key]
        return data

    def normalize_window(self, window):
        head_attributes = [
            "h_x",
            "h_y",
            "elevation",
            "ground_level",
            "ma3"
        ]  # 'in_offset', 'out_offset'
        depth_attributes = ["in_offset", "out_offset"]

        # flow_attributes = ["q_x", "q_y"]

        for atr in head_attributes:
            window = self.depths_normalizer(window, atr)
        for atr in depth_attributes:
            window = self.depths_normalizer(window, atr)
        # for atr in depth_attributes:
        #     window = self.depths_normalizer.scale_attribute(window, atr)

        # for atr in flow_attributes:
        #     if self.abs_flows:
        #         window[atr] = abs(
        #             window[atr]
        #         )  # ! This is done to avoid the direction of the flow given as sign
        #     window = self.flows_normalizer.scale_attribute(window, atr)

        window = self.deltah_normalizer(window,"delta_h")
        window = self.slopeh_normalizer(window, "slope_h")

        window = self.length_normalizer(window, "length")
        window = self.geom_1_normalizer(window, "height")
        window = self.runoff_normalizer(window, "runoff")
        window = self.node_slope_normalizer(window, "node_slope")
        window = self.conduit_slope_normalizer(window, "conduit_slope")
        window = self.volume_normalizer(window, "aprox_conduit_volume")



        window["x"] = window["norm_h_x"]  # * Requirement from PyG
        window["y"] = window["norm_h_y"]  # * Requirement from PyG
        window["edge_attr"] = torch.cat([window["norm_length"],window["norm_height"],
                                         window["norm_conduit_slope"],window["norm_node_slope"],
                                         window["norm_aprox_conduit_volume"],window["norm_in_offset"],
                                         window["norm_out_offset"]],dim=1)
        window["node_attr"] = torch.cat([window["norm_h_x"],window["norm_delta_h"],
                                         window["norm_slope_h"],window["norm_ground_level"],
                                         window["norm_elevation"]],dim=1)
        # window.edge_index


        return window

    def normalize_window_eval(self, window):
        head_attributes = [
            "h_x",
            "elevation",
            "ground_level",
        ]  # 'in_offset', 'out_offset'
        flow_attributes = ["q_x"]

        for atr in head_attributes:
            window = self.depths_normalizer(window, atr)

        for atr in flow_attributes:
            if self.norm_flows:
                window[atr] = abs(
                    window[atr]
                )  # ! This is done to avoid the direction of the flow given as sign
            window = self.flows_normalizer.scale_attribute(window, atr)

        window = self.length_normalizer(window, "length")
        window = self.geom_1_normalizer(window, "height")
        window = self.runoff_normalizer(window, "runoff")
        window = self.node_slope_normalizer(window, "node_slope")
        window = self.conduit_slope_normalizer(window, "conduit_slope")
        window = self.volume_normalizer(window, "aprox_conduit_volume")

        window["x"] = window["norm_h_x"]  # * Requirement from PyG

        return window

    def get_dataloader(self, batch_size):
        list_of_windows = self.get_list_normalized_training_windows()
        return DataLoader(list_of_windows, batch_size)

    def get_unnormalized_heads_pd(self, tensor_heads):
        unnormalized_heads_tensor = self.unnormalize_heads(tensor_heads)
        unnormalized_heads_np = unnormalized_heads_tensor.cpu().detach().numpy()
        # unnormalized_heads_pd = pd.DataFrame(
        #     dict(zip(self.name_nodes, unnormalized_heads_np))
        # )
        # return unnormalized_heads_pd

        return unnormalized_heads_np

    def unnormalize_heads(self, normalized_heads):
        return self.depths_normalizer.unnormalize_attribute(normalized_heads)

    def unscale_flows(self, scaled_flows):
        return self.flows_normalizer.unscale_attribute(scaled_flows)

    def get_unscaled_flows_pd(self, tensor_flows):
        unscaled_flows_tensor = self.unscale_flows(tensor_flows)
        unscaled_flows_np = unscaled_flows_tensor.cpu().detach().numpy()
        unscaled_flows_pd = pd.DataFrame(
            dict(zip(self.name_conduits, unscaled_flows_np))
        )
        return unscaled_flows_pd

    class _InternalNormalizer:

        def __init__(self, parent, attributes):
            self.parent = parent
            self.attributes = attributes

            maxima = torch.zeros(len(self.attributes))
            minima = torch.zeros(len(self.attributes))
            for i, attribute in enumerate(attributes):
                maxima[i] = self.parent._use_function_get_value(self.parent._safe_max, attribute)
                minima[i] = self.parent._use_function_get_value(self.parent._safe_min, attribute)

            self.max_attribute = torch.max(maxima)
            self.min_attribute = torch.min(minima)

        def normalize_attribute(self, window, attribute):
            norm_attribute = self.min_max_normalize(window[attribute])
            window["norm_" + attribute] = norm_attribute.reshape(
                norm_attribute.size()[0], -1
            )
            return window

        def scale_attribute(self, window, attribute):
            norm_attribute = self.min_max_scale(window[attribute])
            window["norm_" + attribute] = norm_attribute.reshape(
                norm_attribute.size()[0], -1
            )
            return window

        def min_max_normalize(self, original_attribute):
            return (original_attribute - self.min_attribute) / (
                self.max_attribute - self.min_attribute
            )

        def unnormalize_attribute(self, attribute):
            return (attribute) * (
                self.max_attribute - self.min_attribute
            ) + self.min_attribute

        def min_max_scale(self, original_attribute):
            return (original_attribute) / (self.max_attribute - self.min_attribute)

        def unscale_attribute(self, original_attribute):
            return (original_attribute) * (self.max_attribute - self.min_attribute)

        def __call__(self, window, attribute):
            return self.normalize_attribute(window, attribute)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"
    class _GlobalZScoreNormalizer:
        def __init__(self, parent, attributes):
            self.parent = parent
            self.attributes = attributes

            self.global_means={}
            self.global_stds={}
            for attr in attributes:
                self.global_means[attr],self.global_stds[attr]=self.parent._use_function_get_zscore(attr)

        def normalize_attribute(self, window,attribute):
            normed = (window[attribute] - self.global_means[attribute]) / (self.global_stds[attribute] + 1e-8)
            window["norm_" + attribute] = normed.reshape(normed.size()[0], -1)
            return window

        def unnormalize(self, normed_tensor, attr):
            return normed_tensor * self.global_stds[attr] + self.global_means[attr]
        def __call__(self, window, attribute):
            return self.normalize_attribute(window, attribute)
        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"

    def _use_function_get_value(self, f, attribute):
        window = self.training_windows[0]
        extreme = f(window[attribute])

        for window in self.training_windows:
            candidate = f(window[attribute])
            extreme = f(torch.stack([extreme, candidate]))
        return extreme
    def _use_function_get_zscore(self,attr):
        all_data = []
        for window in self.training_windows:
            data = window[attr]
            data = data[~torch.isnan(data)]
            if data.numel() > 0:
                all_data.append(data)

        if all_data:
            stacked = torch.cat(all_data)
            mean_value = stacked.mean()
            stds_value = stacked.std(unbiased=False)
        else:
            mean_value = torch.tensor(float('nan'))
            stds_value = torch.tensor(1.0)  # default to avoid division by zero
        return mean_value,stds_value

    def _safe_max(self,tensor):
        tensor = tensor[~torch.isnan(tensor)]
        return torch.max(tensor) if tensor.numel() > 0 else torch.tensor(float('nan'))

    def _safe_min(self,tensor):
        tensor = tensor[~torch.isnan(tensor)]
        return torch.min(tensor) if tensor.numel() > 0 else torch.tensor(float('nan'))

    def __call__(self, window):
        return self.normalize_window(window)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



