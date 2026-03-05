import concurrent.futures
import os
import yaml
from yaml.loader import SafeLoader
import pickle
import numpy as np
import hydroeval as he
import pandas as pd
import networkx as nx
import UtilisSet.Visualize as vis
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from UtilisSet.SWMM_Simulation import SWMMSimulation
from UtilisSet.SWMM_Converter import SWMM_Converter

def NSE_Calcu(pred,true,normalizer,batch_size):
    evaluation = normalizer.get_unnormalized_heads_pd(true)
    simulations = normalizer.get_unnormalized_heads_pd(pred)
    evaluation =evaluation.reshape(batch_size,-1,evaluation.shape[-1])
    simulations = simulations.reshape(batch_size,-1,evaluation.shape[-1])
    nse = np.array([
        1 - np.sum((e[~mask] - s[~mask]) ** 2) / np.sum((e[~mask] - np.mean(e[~mask])) ** 2)
        if np.sum((e[~mask] - np.mean(e[~mask])) ** 2) != 0 else np.nan
        for s, e, mask in zip(simulations, evaluation, np.isnan(simulations) | np.isnan(evaluation))
    ])

    nse_percent = sum((nse >= 0.5).astype(float)) / len(nse)

    return nse_percent
import torch
import matplotlib.pyplot as plt
def plt_true_pred_event(true,pred,node_idx):
    plt.plot(true[:, node_idx], label='true')
    plt.plot(pred[:, node_idx], label='predict')
    plt.legend()
    plt.show()
def plt_true_pred(true,pred):
    plt.plot(true[8, 1, :].reshape(-1), label='true')
    plt.plot(pred[8, 1, :].reshape(-1), label='predict')
    plt.legend()
    plt.show()
def get_mask_input(data,monitored_nodes):
    mask_monitored = torch.full_like(data, fill_value=0).reshape(-1, 436, data.shape[-1])
    monitored_nodes = torch.tensor(monitored_nodes)
    indices = monitored_nodes[torch.randperm(len(monitored_nodes))[:len(monitored_nodes)]]
    mask_monitored[:,indices,:] = 1
    mask_monitored = mask_monitored.reshape(-1, data.shape[-1])
    return data*mask_monitored
def calculate_nse_per_node(pred,true,normalizer,batch_size):
    evaluation = normalizer.get_unnormalized_heads_pd(true)
    simulations = normalizer.get_unnormalized_heads_pd(pred)
    # plt_true_pred(evaluation,simulations)
    evaluation = evaluation.reshape(batch_size,-1,evaluation.shape[-1])
    simulations = simulations.reshape(batch_size,-1,simulations.shape[-1])
    if not isinstance(evaluation, torch.Tensor):
        evaluation = torch.tensor(evaluation, dtype=torch.float32)
    if not isinstance(simulations, torch.Tensor):
        simulations = torch.tensor(simulations, dtype=torch.float32)

    valid_mask = ~torch.isnan(evaluation) & ~torch.isnan(simulations)

    evaluation = torch.where(valid_mask, evaluation, torch.tensor(0.0, device=evaluation.device))
    simulations = torch.where(valid_mask, simulations, torch.tensor(0.0, device=simulations.device))

    valid_count = valid_mask.sum(dim=2, keepdim=True).clamp(min=1)
    mean_obs = evaluation.sum(dim=2, keepdim=True) / valid_count

    numerator = ((evaluation - simulations) ** 2) * valid_mask
    denominator = ((evaluation - mean_obs) ** 2) * valid_mask

    sum_numerator = numerator.sum(dim=2)
    sum_denominator = denominator.sum(dim=2)

    sum_denominator = sum_denominator.clamp(min=1e-6)

    nse = 1 - (sum_numerator / sum_denominator)
    success = (nse > 0.5).float()

    ratio = success.mean(dim=1)
    return ratio
def print_function_name(fn):
    def inner(*args, **kwargs):
        print("{0} executing...".format(fn.__name__))
        to_execute = fn(*args, **kwargs)
        return to_execute

    return inner


def load_yaml(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, encoding='utf-8') as f:
            yaml_data = yaml.load(f, Loader=SafeLoader)
    else:
        raise InvalidYAMLPathException
    return yaml_data


def get_lines_from_textfile(path):
    with open(path, "r") as fh:
        lines = fh.readlines()
    return lines


def get_info_from_file(path):
    info = pd.read_csv(path)
    return info


def get_rain_in_pandas(rain_path):
    rainfall_raw_data = pd.read_csv(rain_path, sep="\t", header=None)
    rainfall_raw_data.columns = [
        "station",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "value",
    ]
    rainfall_raw_data = rainfall_raw_data[
        :-1
    ]
    return rainfall_raw_data

def get_heads_from_file(path):
    # head_raw_data = pd.read_csv(path, index_col=0)  # get_info_from_file(path)
    # head_raw_data = load_zarr(path)
    with open(path, 'rb') as f:
        head_raw_data = pickle.load(f)
    head_raw_data = pd.DataFrame(head_raw_data).T
    # head_raw_data.columns = head_raw_data.columns.str.replace("_Hydraulic_head", "")
    # head_raw_data.columns = head_raw_data.columns.str.replace("node_", "")
    return head_raw_data


def get_flowrate_from_file(path):
    flowrate_raw_data = pd.read_csv(path, index_col=0)

    flowrate_raw_data.columns = flowrate_raw_data.columns.str.replace("_Flow_rate", "")
    flowrate_raw_data.columns = flowrate_raw_data.columns.str.replace("link_", "")
    return flowrate_raw_data


def get_runoff_from_file(path):

    with open(path, 'rb') as f:
        runoff_raw_data = pickle.load(f)
    runoff_raw_data = pd.DataFrame(runoff_raw_data).T

    return runoff_raw_data


def get_dry_periods_index(rainfall_raw_data):

    indexes = np.array(rainfall_raw_data[rainfall_raw_data["value"] == 0].index)
    differences = np.diff(indexes)

    dry_periods_index = []
    single_dry_period_indexes = []
    for i, j in enumerate(differences):
        if j == 1:
            single_dry_period_indexes.append(i)
        else:
            dry_periods_index.append(single_dry_period_indexes)
            single_dry_period_indexes = []

    return dry_periods_index

def extract_simulations_from_file(simulations_path,name_simulation,G):
    hydraulic_heads_path = simulations_path/ "hydraulic_head.pkl"
    # flowrate_raw_path = simulations_path / name_simulation / "flow_rate.csv"
    runoff_path = simulations_path / "runoff.pkl"
    # rain_path = simulations_path / name_simulation / name_simulation+'.dat'

    heads_raw_data = get_heads_from_file(hydraulic_heads_path)
    # flowrate_raw_data = get_flowrate_from_file(flowrate_raw_path)
    runoff_raw_data = get_runoff_from_file(runoff_path)

    nodes_subcatchment = nx.get_node_attributes(G, "subcatchment")

    reversed_nodes_subcatchment = {
        value: key for key, value in nodes_subcatchment.items()
    }

    runoff_raw_data = runoff_raw_data.rename(columns=reversed_nodes_subcatchment)
    missing_nodes = set(heads_raw_data.columns) - set(runoff_raw_data.columns)

    for i in missing_nodes:
        runoff_raw_data[i] = 0

    # rain_raw_data = get_rain_in_pandas(rain_path)
    raw_data = {
        "heads_raw_data": heads_raw_data,
        "runoff_raw_data": runoff_raw_data,
    }
    # "rain_raw_data":rain_raw_data}
    sim = SWMMSimulation(G, raw_data, name_simulation)
    # sim = [G, raw_data, name_simulation]
    return sim
def extract_simulations_from_folders(simulations_path, inp_path, max_events=-1):
    list_of_simulations = os.listdir(simulations_path)


    if max_events == -1:
        max_events = len(list_of_simulations)

    converter = SWMM_Converter(inp_path, is_directed=True)
    G = converter.inp_to_G()
    simulations = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(extract_simulations_from_file, simulations_path/sim_path,sim_path,G)
            for sim_path in list_of_simulations
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                simulations.append(result)
            except Exception as e:
                print("提取任务失败:", e)

    return simulations


def get_all_windows_from_list_simulations(simulations, steps_behind, steps_ahead):
    windows = []
    for sim in simulations:
        windows += sim.get_all_windows(
            steps_behind=steps_behind, steps_ahead=steps_ahead
        )
    return windows


def save_pickle(variable, path):
    with open(path, "wb") as handle:
        pickle.dump(variable, handle)


def load_pickle(path):
    with open(path, "rb") as handle:
        variable = pickle.load(handle)
    return variable


def run_model_in_validation_event(ml_experiment, model, event_index=5):

    val_event = ml_experiment.validation_simulations[event_index]
    normalizer = ml_experiment.normalizer
    steps_behind = ml_experiment.steps_behind

    sim_in_window = val_event.get_simulation_in_one_window(steps_behind)
    norm_sim_in_window = normalizer.normalize_window(sim_in_window)

    swmm_heads_pd = normalizer.get_unnormalized_heads_pd(norm_sim_in_window["norm_h_y"])

    predicted_heads_pd = normalizer.get_unnormalized_heads_pd(model(norm_sim_in_window))
    return swmm_heads_pd, predicted_heads_pd


def r2_median_wet_weather(swmm_heads_pd, predicted_heads_pd, elevation):

    mask_wet = abs(swmm_heads_pd - elevation) > 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_median(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_median_dry_weather(swmm_heads_pd, predicted_heads_pd, elevation):
    mask_dry = abs(swmm_heads_pd - elevation) < 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_dry]
    masked_predicted_heads_pd = predicted_heads_pd[mask_dry]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_median(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_wet_weather(swmm_heads_pd, predicted_heads_pd, elevation):

    mask_wet = abs(swmm_heads_pd - elevation) > 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_score(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_dry_weather(swmm_heads_pd, predicted_heads_pd, elevation):

    mask_dry = abs(swmm_heads_pd - elevation) < 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_dry]
    masked_predicted_heads_pd = predicted_heads_pd[mask_dry]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_score(
        pd.DataFrame(masked_swmm_heads), pd.DataFrame(masked_predicted_heads)
    )


def r2_flow_wet_weather(swmm_flows_pd, predicted_heads_pd):

    mask_wet = abs(swmm_flows_pd) > 0.00001

    masked_swmm_heads_pd = swmm_flows_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    masked_swmm_heads = masked_swmm_heads_pd.to_numpy()
    masked_swmm_heads = masked_swmm_heads[~np.isnan(masked_swmm_heads)]

    masked_predicted_heads = masked_predicted_heads_pd.to_numpy()
    masked_predicted_heads = masked_predicted_heads[~np.isnan(masked_predicted_heads)]

    return r2_score(masked_swmm_heads, masked_predicted_heads)


def r2_flow_dry_weather(swmm_flows_pd, predicted_flows_pd):

    mask_dry = abs(swmm_flows_pd) < 0.00001

    masked_swmm_flows_pd = swmm_flows_pd[mask_dry]
    masked_predicted_flows_pd = predicted_flows_pd[mask_dry]

    masked_swmm_flows = masked_swmm_flows_pd.to_numpy()
    masked_swmm_flows = masked_swmm_flows[~np.isnan(masked_swmm_flows)]

    masked_predicted_flows = masked_predicted_flows_pd.to_numpy()
    masked_predicted_flows = masked_predicted_flows[~np.isnan(masked_predicted_flows)]

    return r2_score(masked_swmm_flows, masked_predicted_flows)


def r2_overall(swmm_variable_pd, predicted_variable_pd, *args, **kwargs):

    flattened_swmm = swmm_variable_pd.to_numpy().flatten()
    flattened_prediction = predicted_variable_pd.to_numpy().flatten()
    return r2_score(flattened_swmm, flattened_prediction)


def wet_r2_per_node(swmm_heads_pd, predicted_heads_pd, elevation):

    mask_wet = abs(swmm_heads_pd - elevation) > 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_wet]
    masked_predicted_heads_pd = predicted_heads_pd[mask_wet]

    wet_r2 = []
    for i in range(len(masked_swmm_heads_pd.columns)):
        try:
            r2 = r2_score(
                masked_swmm_heads_pd.iloc[:, i].dropna().to_numpy(),
                masked_predicted_heads_pd.iloc[:, i].dropna().to_numpy(),
            )
        except Exception as e:
            r2 = -9.99
        r2 = np.clip(r2, -10, 1)
        wet_r2.append(r2)
    return wet_r2


def dry_r2_per_node(swmm_heads_pd, predicted_heads_pd, elevation):

    mask_dry = abs(swmm_heads_pd - elevation) < 0.00001

    masked_swmm_heads_pd = swmm_heads_pd[mask_dry]
    masked_predicted_heads_pd = predicted_heads_pd[mask_dry]

    dry_r2 = []
    for i in range(len(masked_swmm_heads_pd.columns)):
        try:
            r2 = r2_score(
                masked_swmm_heads_pd.iloc[:, i].dropna().to_numpy(),
                masked_predicted_heads_pd.iloc[:, i].dropna().to_numpy(),
            )
        except Exception as e:
            r2 = -9.99

        r2 = np.clip(r2, -10, 1)
        dry_r2.append(r2)

    return dry_r2


def r2_median(swmm_variable_pd, predicted_variable_pd, *args, **kwargs):

    r2s = []
    for i in range(len(swmm_variable_pd.columns)):
        r2s.append(
            r2_score(
                swmm_variable_pd.iloc[:, i].to_numpy(),
                predicted_variable_pd.iloc[:, i].to_numpy(),
            )
        )
    return np.median(r2s)


def r2_mean(swmm_variable_pd, predicted_variable_pd, *args, **kwargs):

    r2s = []
    for i in range(len(swmm_variable_pd.columns)):
        r2s.append(
            r2_score(
                swmm_variable_pd.iloc[:, i].to_numpy(),
                predicted_variable_pd.iloc[:, i].to_numpy(),
            )
        )
    return np.mean(r2s)


class InvalidYAMLPathException(Exception):
    pass
