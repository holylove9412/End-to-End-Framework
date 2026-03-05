import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
import math
class SWMMSimulation:
    def __init__(self, G, raw_data, name_simulation):

        self.G = G
        self.heads_raw_data = raw_data["heads_raw_data"]
        self.delta_head_raw_data = self.heads_raw_data.diff().bfill()
        time_diff = (self.heads_raw_data.index[1]-self.heads_raw_data.index[0]).total_seconds()/60 ##获取时间间隔
        self.slope_head_raw_data = self.delta_head_raw_data/time_diff
        # self.flowrate_raw_data = raw_data["flowrate_raw_data"]
        self.runoff_raw_data = raw_data["runoff_raw_data"]
        # self.rain_raw_data      = raw_data['rain_raw_data']
        self.name_simulation = name_simulation

        self.simulation_length = len(self.runoff_raw_data)

    def get_simulation_in_one_window(self, steps_behind, is_training=True):
        windows = self.get_all_windows(
            steps_behind=steps_behind,
            steps_ahead=12,
            is_training=is_training,
        )
        # assert len(windows) == 1, "There should be one and only one window."
        # one_window = windows[0]

        return windows

    def get_all_windows(self, steps_behind, steps_ahead, is_training=True):
        assert steps_ahead > 0, "The steps ahead  should be greater than 0"
        assert steps_behind > 0, "The steps behind should be greater than 0"

        windows_list = []
        for time in range(0,self.simulation_length-steps_behind+1,1):
            window = self.get_window(
                steps_behind, steps_ahead, time, is_training=is_training
            )
            windows_list.append(window)

        return windows_list

    def get_window(self, steps_behind, steps_ahead, time, is_training=True):
        # self._checkOutOfBounds(steps_behind, steps_ahead, time)

        h0 = self._get_h0_for_window(time, steps_behind)
        # q0 = self._get_q0_for_window(time, steps_behind)
        ro_timeperiod = self._get_ro_for_window(time, steps_behind)

        delta_h = self._get_deltah_for_window(time,steps_behind) ##水位变化量

        slope_h = self._get_slopeh_for_window(time,steps_behind) ##水位变化率

        ma3 = self._get_ma3_for_window(time,steps_behind) ##移动平均

        delta_h_x_dict = self._get_features_dictionary(delta_h)
        slope_h_x_dict = self._get_features_dictionary(slope_h)
        ma3_x_dict = self._get_features_dictionary(ma3)

        h_x_dict = self._get_features_dictionary(h0)
        # q_x_dict_conduit = self._get_features_dictionary(q0)
        # q_x_dict = self._change_conduit_name_to_edge_tuple(q_x_dict_conduit)
        ro_x_dict = self._get_features_dictionary(ro_timeperiod)

        G_for_window = self.G.to_undirected()

        nx.set_node_attributes(G_for_window, h_x_dict, name="h_x")
        nx.set_node_attributes(G_for_window, ro_x_dict, name="runoff")

        nx.set_node_attributes(G_for_window,delta_h_x_dict, name="delta_h")
        nx.set_node_attributes(G_for_window, slope_h_x_dict, name="slope_h")
        nx.set_node_attributes(G_for_window, ma3_x_dict, name="ma3")

        # nx.set_edge_attributes(G_for_window, q_x_dict, name="q_x")

        if is_training:
            ht_timeperiod = self._get_ht_for_window(time, steps_behind,steps_ahead)
            # ht_timeperiod = self._get_ht_for_window(time, steps_ahead)
            # qt_timeperiod = self._get_qt_for_window(time, steps_ahead)

            h_y_dict = self._get_features_dictionary(ht_timeperiod)
            # q_y_dict_conduit = self._get_features_dictionary(qt_timeperiod)
            # q_y_dict = self._change_conduit_name_to_edge_tuple(q_y_dict_conduit)

            nx.set_node_attributes(G_for_window, h_y_dict, name="h_y")
            # nx.set_edge_attributes(G_for_window, q_y_dict, name="q_y")


        window = from_networkx(G_for_window)

        window["ground_level"] = window["elevation"] + window["max_depth"]


        geometric_attributes = zip(
            np.array(window.height), np.array(window.length)
        )
        aprox_volume = torch.tensor(
            list(map(self._get_aprox_volume, geometric_attributes)), dtype=torch.float32
        )
        src, dst = window.edge_index
        node_slope = (window.elevation[dst] - window.elevation[src]) / window.length
        conduit_slope_ = (window.in_offset[dst] - window.out_offset[src]) / window.length
        is_conduit=window.is_pump == 0
        conduit_slope = torch.full_like(conduit_slope_,fill_value=float('nan'))
        conduit_slope[is_conduit] = conduit_slope_[is_conduit]

        window["aprox_conduit_volume"] = aprox_volume
        window["node_slope"] = node_slope
        window["conduit_slope"] = conduit_slope
        # print(type(window["conduit_slope"]))

        window["steps_ahead"] = steps_ahead
        window["steps_behind"] = steps_behind

        return window

    def _get_aprox_volume(self, geoms):
        try:
            diameter, length = geoms
            return (1 / 4) * math.pi * (diameter ** 2) * length
        except (TypeError, ValueError):
            return np.nan

    def _change_conduit_name_to_edge_tuple(self, q_x_dict_conduit):
        return {
            self.G.graph["conduit_phonebook"][k]: value
            for k, value in q_x_dict_conduit.items()
        }

    def _checkOutOfBounds(self, steps_behind, steps_ahead, time):
        max_allowable_time = self.simulation_length - steps_ahead
        if time > max_allowable_time:
            raise AfterEndTimeException
        if time - (steps_behind - 1) < 0:
            raise BeforeZeroTimeException

    def _get_h0_for_window(self, time, steps_behind):
        # lagged_time = time - (steps_behind - 1)
        return self.heads_raw_data.iloc[time : time + steps_behind, :]
    def _get_deltah_for_window(self, time, steps_behind):
        # lagged_time = time - (steps_behind - 1)
        return self.delta_head_raw_data.iloc[time : time + steps_behind, :]
    def _get_slopeh_for_window(self, time, steps_behind):
        return self.slope_head_raw_data.iloc[time : time + steps_behind, :]
    def _get_ma3_for_window(self, time, steps_behind):
        self.ma3 = self.heads_raw_data.rolling(window=steps_behind,min_periods=1).mean()
        return self.ma3.iloc[time : time + steps_behind, :]

    def _get_h0_for_window_tensor(self, time, steps_behind):
        lagged_time = time - (steps_behind - 1)
        tensor_h0 = torch.tensor(
            self.heads_raw_data.iloc[lagged_time : time + 1, :].values,
            dtype=torch.float32,
        ).t()
        return tensor_h0

    def _get_q0_for_window(self, time, steps_behind):
        lagged_time = time - (steps_behind - 1)
        return self.flowrate_raw_data.iloc[lagged_time : time + 1, :]

    def _get_q0_for_window_tensor(self, time, steps_behind):
        lagged_time = time - (steps_behind - 1)
        tensor_q0 = torch.tensor(
            self.flowrate_raw_data.iloc[lagged_time : time + 1, :].values,
            dtype=torch.float32,
        ).t()
        return tensor_q0

    def _get_ro_for_window(self, time,  steps_behind):
        # lagged_time = time - (steps_behind - 1)
        return self.runoff_raw_data.iloc[time :time + steps_behind, :]

    def _get_ro_for_window_tensor(self, time, steps_ahead, steps_behind):
        lagged_time = time - (steps_behind - 1)
        tensor_runoff = torch.tensor(
            self.runoff_raw_data.iloc[lagged_time : time + 1 + steps_ahead, :].values,
            dtype=torch.float32,
        ).t()
        return tensor_runoff

    def _get_ht_for_window(self, time, steps_behind,steps_ahead):
        return self.heads_raw_data.iloc[time + steps_behind-steps_ahead : time + steps_behind, :]

    def _get_ht_for_window_tensor(self, time, steps_ahead):
        return torch.tensor(
            self.heads_raw_data.iloc[time + 1 : time + 1 + steps_ahead, :].values,
            dtype=torch.float32,
        ).t()

    def _get_qt_for_window(self, time, steps_ahead):
        return self.flowrate_raw_data.iloc[time + 1 : time + 1 + steps_ahead, :]

    def _get_qt_for_window_tensor(self, time, steps_ahead):
        return torch.tensor(
            self.flowrate_raw_data.iloc[time + 1 : time + 1 + steps_ahead, :].values,
            dtype=torch.float32,
        ).t()

    def _get_features_dictionary(self, *args):
        features_df = pd.concat(args).reset_index(drop=True).transpose()
        node_names = list(features_df.index)
        list_features = features_df.values.tolist()
        input_features_dict = dict(zip(node_names, list_features))

        return input_features_dict

    def nx_node_attribute_to_tensor(self, attribute):
        values = list(nx.get_node_attributes(self.G, attribute).values())
        return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

    def nx_edge_attribute_to_tensor(self, attribute):
        values = list(nx.get_edge_attributes(self.G, attribute).values())
        return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name_simulation})"

    def __len__(self):
        return self.simulation_length


class BeforeZeroTimeException(Exception):
    pass


class AfterEndTimeException(Exception):
    pass
