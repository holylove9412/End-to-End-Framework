import sys

sys.path.insert(0, "")
import wandb
import random
import numpy as np
import re
import torch.optim as optim
from UtilisSet.Trainer import TrainerFactory
from torch_geometric.loader import DataLoader
from UtilisSet.Reconstruction_models.Model_main import ModelFactory
from UtilisSet.MetricCalculator import MetricCalculator
from UtilisSet.Normalizer import NormalizerFactory
from pathlib import Path

import UtilisSet.Visualize as vis
import os
import UtilisSet.utils as utils
import hydroeval as he
import torch
import torch.nn as nn

class Backbone:
    def __init__(self, config, data_dir, saved_objects_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.data_dir = data_dir
        self.saved_objects_dir = saved_objects_dir
        self._extract_hyperparams_from_config()
        self._define_data_paths()
        self._set_up_experiment()
        self.metrics_calculator = MetricCalculator()
        self.nodes_dashboard = None

    def _set_up_experiment(self):
        self._set_up_seeds()
        # self._set_up_simulations()
        # self._set_up_windows()
        self._set_up_normalizer()
        self._set_up_normalized_windows()
        self._set_up_dataloaders()
        self._set_up_model()
        self._set_up_trainer()

    def train_model(self):
        self.trainer.train(self.training_loaders, self.validation_loader,self.epochs)

    def run_full_experiment(self):
        self._profile_model_with_multiple_events()
        self.train_model()
        self.use_model_in_validation_events()
        self.finish_ML_tracker()

    def use_model_in_validation_events(self):
        val_event_index = 1  # 9
        val_event = self.validation_simulations[val_event_index]
        sim_in_window = val_event.get_simulation_in_one_window(self.steps_behind)
        norm_sim_in_window = self.normalizer.normalize_window(sim_in_window).to(
            self.device
        )
        name_event = val_event.name_simulation

        predicted_heads_pd = self.normalizer.get_unnormalized_heads_pd(
            self.model(norm_sim_in_window)
        )

        swmm_heads_pd = self.normalizer.get_unnormalized_heads_pd(
            norm_sim_in_window["norm_h_y"]
        )

        nodes_to_graph = self.nodes_to_plot
        for node in nodes_to_graph:
            self._save_response_graphs_in_ML_tracker(
                swmm_heads_pd,
                predicted_heads_pd,
                name_event,
                node,
                type_variable="Head",
            )


    def change_model(self, new_model):
        self.model = new_model
        self._set_up_trainer()

    def change_normalizer(self, new_normalizer):
        self.normalizer = new_normalizer

        self.normalized_training_windows = [
            [self.normalizer.normalize_window(tra_win) for tra_win in tra_level]
            for tra_level in self.training_windows
        ]
        self.normalized_validation_windows = [
            self.normalizer.normalize_window(val_win)
            for val_win in self.validation_windows
        ]

    @utils.print_function_name

    def _extract_hyperparams_from_config(self):
        self.trainer_name = self.config["trainer_name"]
        self.node_loss_weight = self.config["node_loss_weight"]
        self.edge_loss_weight = self.config["edge_loss_weight"]
        self.use_pre_trained_weights = self.config["use_pre_trained_weights"]
        self.use_trained_weights = self.config["use_trained_weights"]
        self.requires_freezing = self.config["requires_freezing"]

        self.abs_flows = self.config["abs_flows"]

        self.num_events_training = self.config["num_events_training"]
        self.num_events_validation = self.config["num_events_validation"]
        self.balance_ratio = self.config["balance_ratio"]
        self.variance_threshold = self.config["variance_threshold"]

        self.edge_input_list = self.config["edge_input_list"].split(",")
        self.base_model_name = self.config["base_model_name"]
        self.model_name = self.config["model_name"]
        self.n_hidden_layers = self.config["n_hidden_layers"]
        self.n_gcn_layers = self.config["n_gcn_layers"]
        self.non_linearity = self.config["non_linearity"]
        self.normalizer_name = self.config["normalizer_name"]
        self.steps_behind = self.config["steps_behind"]
        self.steps_ahead = self.config["steps_ahead"]
        self.steps_ahead_validation = self.config["steps_ahead_validation"]
        self.prediction_steps = self.config["prediction_steps"]
        self.hidden_dim = self.config["hidden_dim"]
        self.skip_alpha = self.config["skip_alpha"]

        self.monitored_nodes = self.config["monitored_nodes"]
        # self.num_monitored_nodes = len(self.config["monitored_nodes"])
        # self.monitored_nodes = random.sample(list(range(436)),48)
        # self.monitored_nodes = random.sample(list(range(362)),int(362*0.1))

        self.epochs = self.config["epochs"]
        self.batch_size = self.config["batch_size"]
        self.learning_rate = self.config["learning_rate"]
        self.weight_decay = self.config["weight_decay"]
        self.gamma_scheduler = self.config["gamma_scheduler"]
        self.gamma_loss = self.config["gamma_loss"]
        self.switch_epoch = self.config["switch_epoch"]
        self.min_expected_loss = self.config["min_expected_loss"]
        self.use_saved_normalizer = self.config["use_saved_normalizer"]
        self.normalizer_name = self.config["normalizer_name"]
        self.saved_normalizer_name = self.config["saved_normalizer_name"]

        self.use_training_simulations = self.config["use_training_simulations"]
        self.use_validation_simulations = self.config["use_validation_simulations"]
        self.use_testing_simulations = self.config["use_testing_simulations"]

        self.use_saved_normalized_windows_dataset = self.config["use_saved_normalized_windows_dataset"]
        self.saved_normalized_training_dataset_name = self.config["saved_normalized_training_dataset_name"]
        self.saved_normalized_validation_dataset_name = self.config["saved_normalized_validation_dataset_name"]

        self.seed = self.config["seed"]

        self.k_hops = self.config["k_hops"]
        self.eps_gnn = self.config["eps_gnn"]

        self.nodes_to_plot = self.config["nodes_to_plot"]

    def _define_data_paths(self):
        self.training_simulations_path = (
            Path(self.data_dir) / self.config["network"] / "simulations" / "training"
        )
        self.validation_simulations_path = (
            Path(self.data_dir) / self.config["network"] / "simulations" / "validation"
        )
        self.testing_simulations_path = (
            Path(self.data_dir) / self.config["network"] / "simulations" / "testing"
        )
        self.inp_path = (
            Path(self.data_dir)
            / self.config["network"]
            / "networks"
            / "".join([self.config["network"], ".inp"])
        )
        self.pre_trained_weights_path = (
            Path(self.saved_objects_dir)
            / "saved_weights"
            / self.config["pre_trained_weights"]
        )

        self.trained_weights_path = (
            Path(self.saved_objects_dir)
            / "saved_weights"
            / self.config["trained_weights"]
        )


    @utils.print_function_name
    def _set_up_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed + 1)
        torch.manual_seed(self.seed + 2)
        # torch.use_deterministic_algorithms(True, warn_only = True)

    @utils.print_function_name
    def _set_up_simulations(self):
        self.training_simulations, self.validation_simulations,self.testing_simulations = (
            self._read_simulation_data()
        )

    @utils.print_function_name
    def _set_up_windows(self):
        self.training_windows = self._get_training_windows()
        self.validation_windows = self._get_validation_windows()

    @utils.print_function_name
    def _set_up_normalizer(self):
        saved_normalizers_path = Path(self.saved_objects_dir) / "saved_normalizers"
        if self.use_saved_normalizer:

            self.normalizer = utils.load_pickle(
                saved_normalizers_path
                / self.saved_normalizer_name
            )
            print("Using saved normalizer: ", self.saved_normalizer_name)

            # self.normalizer.name_nodes = self.validation_windows[0].name_nodes

        else:
            self.normalizer = NormalizerFactory(self.normalizer_name)(
                self.training_windows[0], abs_flows=self.abs_flows
            )

            saved_normalizers_path.mkdir(parents=True, exist_ok=True)
            utils.save_pickle(self.normalizer, (saved_normalizers_path / self.saved_normalizer_name))


    @utils.print_function_name
    def _set_up_normalized_windows(self):
        saved_normalizerd_widonws_path = Path(self.saved_objects_dir) / "saved_normalized_windows"
        if self.use_saved_normalized_windows_dataset:
            self.normalized_training_windows = torch.load(saved_normalizerd_widonws_path / self.saved_normalized_training_dataset_name)
            self.normalized_validation_windows = torch.load(saved_normalizerd_widonws_path / self.saved_normalized_validation_dataset_name)

        else:
            self.normalized_training_windows = [
                [self.normalizer.normalize_window(tra_win) for tra_win in tra_level]
                for tra_level in self.training_windows
            ]
            self.normalized_validation_windows = [
                self.normalizer.normalize_window(val_win)
                for val_win in self.validation_windows
            ]

            # keep_fields = ["x", "y", "edge_index", "norm_h_x", "norm_h_y", "norm_ground_level", "steps_ahead",
            #                "steps_behind", "name_nodes",
            #                "norm_elevation", "norm_in_offset", "norm_out_offset", "norm_length", "norm_height",
            #                "norm_runoff", "norm_node_slope", "norm_conduit_slope"]
            keep_fields = ["x", "y", "edge_index", "norm_h_x", "norm_h_y",  "steps_ahead","steps_behind", "name_nodes",
                           "norm_runoff","edge_attr","node_attr","is_pump"]
            # keep_fields = ["x", "y", "edge_index", "norm_runoff", "edge_attr", "node_attr"]
            self.extract_normalized_training_windows = [
                [self.normalizer.prune_data(tra_win,keep_fields) for tra_win in tra_level]
                for tra_level in self.normalized_training_windows
            ]

            self.extract_normalized_validation_windows = [
                self.normalizer.prune_data(val_win,keep_fields)
                for val_win in self.normalized_validation_windows
            ]

            torch.save(self.extract_normalized_training_windows,
                       saved_normalizerd_widonws_path / self.saved_normalized_training_dataset_name)
            torch.save(self.extract_normalized_validation_windows,
                   saved_normalizerd_widonws_path / self.saved_normalized_validation_dataset_name)


    @utils.print_function_name
    def _set_up_dataloaders(self):

        self.training_loaders = [
            DataLoader(
                norm_tra_level, batch_size=self.batch_size, num_workers=8,pin_memory=True,shuffle=True, drop_last=False
            )
            for norm_tra_level in self.normalized_training_windows
        ]
        self.validation_loader = DataLoader(
            self.normalized_validation_windows,num_workers=8,pin_memory=True,
            batch_size=self.batch_size,shuffle=True,
            drop_last=False,
        )

    @utils.print_function_name
    def _set_up_model(self):
        if self.use_pre_trained_weights:
            self.base_model = ModelFactory(self.base_model_name)(
                steps_behind=self.steps_behind,
                hidden_dim=self.hidden_dim,
                skip_alpha=self.skip_alpha,
                k_hops=self.k_hops,
                eps_gnn=self.eps_gnn,
                n_hidden_layers=self.n_hidden_layers,
                n_gcn_layers=self.n_gcn_layers,
                non_linearity=self.non_linearity,
                edge_input_list=self.edge_input_list,
            )
            self.base_model.to(self.device)
            self.assess_pre_training_weights()
            self.model = ModelFactory(self.model_name)(
                base_model=self.base_model,
                steps_behind=self.steps_behind,
                hidden_dim=self.hidden_dim,
                skip_alpha=self.skip_alpha,
                k_hops=self.k_hops,
                eps_gnn=self.eps_gnn,
                n_hidden_layers=self.n_hidden_layers,
                n_gcn_layers=self.n_gcn_layers,
                non_linearity=self.non_linearity,
                edge_input_list=self.edge_input_list,
            )
            self.model.to(self.device)
            self.assess_training_weights()
        else:
            self.model = ModelFactory(self.model_name)(
                steps_behind=self.steps_behind,
                hidden_dim=self.hidden_dim,
                skip_alpha=self.skip_alpha,
                k_hops=self.k_hops,
                eps_gnn=self.eps_gnn,
                n_hidden_layers=self.n_hidden_layers,
                n_gcn_layers=self.n_gcn_layers,
                non_linearity=self.non_linearity,
                edge_input_list=self.edge_input_list,
            )
            self.model.to(self.device)
            self.assess_training_weights()

    def assess_pre_training_weights(self):
        if self.use_pre_trained_weights:
            saved_weights_dict = torch.load(
                self.pre_trained_weights_path, map_location="cpu"
            )
            self.base_model.load_state_dict(saved_weights_dict['model_state_dict'], strict=True)
            print("Using pre-trained weights:", self.config["pre_trained_weights"])
            if self.requires_freezing:
                for parameter_name, param in self.base_model.named_parameters():
                    if parameter_name in saved_weights_dict.keys():
                        param.requires_grad = False
    def assess_training_weights(self):
        if self.use_trained_weights:
            saved_weights_dict = torch.load(
                self.trained_weights_path, map_location="cpu"
            )
            self.model.load_state_dict(saved_weights_dict['model_state_dict'], strict=True)
            print("Using trained weights:", self.config["trained_weights"])
            if self.requires_freezing:
                for parameter_name, param in self.model.named_parameters():
                    if parameter_name in saved_weights_dict.keys():
                        param.requires_grad = False

    @utils.print_function_name
    def _set_up_trainer(self):
        if self.use_pre_trained_weights:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.use_trained_weights:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.gamma_scheduler
        )

        self.criterion = nn.MSELoss()

        chosen_trainer = TrainerFactory(self.trainer_name)
        self.trainer = chosen_trainer(
            self.model,
            self.optimizer,
            self.criterion,
            self.scheduler,
            report_freq=2,
            trained_weights_path=self.trained_weights_path,
            switch_epoch=self.switch_epoch,
            min_expected_loss=self.min_expected_loss,
            node_loss_weight=self.node_loss_weight,
            edge_loss_weight=self.edge_loss_weight,
            monitored_nodes=self.monitored_nodes,
            normalizer=self.normalizer,
            num_nodes=len(self.normalized_training_windows[0][0]['name_nodes'])
        )

    def _read_simulation_data(self):

        saved_simulation_path = Path(self.saved_objects_dir) / "saved_simulations"
        if self.use_training_simulations:
            training_simulations = utils.load_pickle(
                saved_simulation_path
                / 'training_simulations.pk'
            )
            validation_simulations = utils.load_pickle(
                saved_simulation_path
                / 'validation_simulations.pk'
            )
            testing_simulations = utils.load_pickle(
                saved_simulation_path
                / 'testing_simulations.pk'
            )
        else:
            training_simulations = utils.extract_simulations_from_folders(self.training_simulations_path,
                                                            self.inp_path, self.num_events_training)
            validation_simulations = utils.extract_simulations_from_folders(self.validation_simulations_path,
                                                            self.inp_path,  self.num_events_validation)
            testing_simulations = utils.extract_simulations_from_folders(self.testing_simulations_path,
                                                            self.inp_path,  50)
            utils.save_pickle(training_simulations, (saved_simulation_path / 'training_simulations.pk'))
            utils.save_pickle(validation_simulations, (saved_simulation_path / 'validation_simulations.pk'))
            utils.save_pickle(testing_simulations, (saved_simulation_path / 'testing_simulations.pk'))

        return training_simulations, validation_simulations,testing_simulations

    def _get_training_windows(self):
        training_windows_names = self.config["training_windows_names"]
        use_saved_training_windows = self.config["use_saved_training_windows"]

        folder_path = (
            Path(self.saved_objects_dir) / "saved_windows" / self.config["network"]
        )
        if use_saved_training_windows:
            training_windows = []
            for path in training_windows_names:
                training_windows.append(utils.load_pickle(folder_path / path))
                print("Using loaded windows: ", path)

        else:
            training_windows = []
            for step_ahead in self.steps_ahead:
                instance_windows = self._get_balanced_windows_from_list_simulations(
                    self.training_simulations, self.steps_behind, step_ahead
                )
                training_windows.append(instance_windows)
            for variable,path in zip(training_windows,training_windows_names):
                folder_path.mkdir(parents=True, exist_ok=True)
                utils.save_pickle(variable,(folder_path / path))

        return training_windows

    def _get_validation_windows(self):
        validation_windows_name = self.config["validation_windows_name"]
        use_saved_validation_windows = self.config["use_saved_validation_windows"]

        folder_path = (
            Path(self.saved_objects_dir) / "saved_windows" / self.config["network"]
        )
        validation_windows_path = folder_path / validation_windows_name

        if use_saved_validation_windows:
            validation_windows = utils.load_pickle(validation_windows_path)
            print("Using loaded windows: ", validation_windows_name)
        else:
            validation_windows = self._get_balanced_windows_from_list_simulations(
                self.validation_simulations,
                self.steps_behind,
                self.steps_ahead_validation
            )
            utils.save_pickle(validation_windows, (folder_path / validation_windows_name))
        return validation_windows

    def _load_simulation_object(self, sim_id,mode='training'):
        if mode=='training':
            sim = self.training_simulations[sim_id]
        else:
            sim = self.validation_simulations[sim_id]
        return sim
    def _extract_balanced_from_sim(self,sim,steps_behind,steps_ahead):
        windows=[]
        static=[]
        heads = sim.heads_raw_data
        simulation_length = len(heads)
        length_window = steps_ahead + steps_behind
        max_time_allowed = simulation_length - steps_ahead

        for t in range(0,simulation_length-steps_behind+1,1):
            if heads.iloc[t: t + steps_behind].var().max() > self.variance_threshold:
                window = sim.get_window(steps_behind, steps_ahead, t)
                windows.append(window)
            else:
                if len(static) == 0:
                    window = sim.get_window(steps_behind, steps_ahead, t)
                    static.append(window)

        return windows,static

    def _sliding_window_split(self,data, window_size, step=1):
        windows = []
        for i in range(0, len(data) - window_size + 1, step):
            windows.append(data[i:i + window_size])
        return windows

    def _get_balanced_windows_from_list_simulations(
        self, list_simulations, steps_behind, steps_ahead
    ):

        windows_list = []
        static_window = []
        for sim in list_simulations:
            win, static = self._extract_balanced_from_sim(sim, steps_behind, steps_ahead)
            windows_list.extend(win)
            if static:
                static_window.append(static[0])

        windows = windows_list
        return windows
    def get_testing_loader(self):
        saved_loader_path = Path(self.saved_objects_dir) / "saved_normalized_testing_windows"
        for event_index in range(len(self.testing_simulations)):
            test_event = self.testing_simulations[event_index]
            normalizer = self.normalizer
            steps_behind = self.steps_behind

            sim_in_window_list = test_event.get_simulation_in_one_window(steps_behind)

            norm_sim_in_window = [normalizer.normalize_window(sim_in_window) for sim_in_window in sim_in_window_list]

            keep_fields = ["x", "y", "edge_index", "norm_runoff","edge_attr","node_attr"]
            extract_normalized_windows = [self.normalizer.prune_data(tra_win,keep_fields) for tra_win in norm_sim_in_window]

            utils.save_pickle(extract_normalized_windows, (saved_loader_path / f'{test_event}.pk'))
    def one_event_run_in_model(self,data_path,event):
        norm_in_window = utils.load_pickle(
            data_path
            / event
        )
        name = re.findall(r'\((.*?)\)', event)
        event_loader = DataLoader(norm_in_window, batch_size=self.batch_size, num_workers=8, pin_memory=True,
                                      shuffle=False, drop_last=False)
        yhat_list =[]
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for val_list in event_loader:
                val_list['x'] = utils.get_mask_input(val_list.x,self.monitored_nodes)
                val_list['node_attr'] = torch.cat([utils.get_mask_input(val_list.node_attr[:,:72], self.monitored_nodes),
                                                   val_list.node_attr[:,-2:]],dim=1)
                yhat = self.model(val_list.to(self.device))[:,-12:]
                yhat_list.append(yhat.view(-1,len(self.normalized_training_windows[0][0]['name_nodes']),yhat.size(-1)))

        y_heads = torch.cat(yhat_list,dim=0)

        target_heads = torch.stack([real_y["y"]for real_y in norm_in_window],dim=0)

        swmm_heads_pd = self.normalizer.get_unnormalized_heads_pd(target_heads)
        predicted_heads_pd = self.normalizer.get_unnormalized_heads_pd(y_heads)
        nse = np.array([he.evaluator(he.nse, predicted_heads_pd[:,:,0][:,i], swmm_heads_pd[:,:,0][:,i]) for i in range(len(self.normalized_training_windows[0][0]['name_nodes']))]).reshape(-1)
        # ng = Neighbour_nodes()
        # us_node,ds_node=ng.get_neighbour_nodes(298,2)
        vis.nx_plot(self.monitored_nodes, nse, self.inp_path)
        # utils.plt_true_pred_event(swmm_heads_pd[:,:,0],predicted_heads_pd[:,:,0],3)
        nse_percent_5 = sum((nse >= 0.5).astype(float)) / len(nse)
        nse_percent_0 = sum((nse >= 0).astype(float)) / len(nse)
        print("Event: %s >>>NSE>0.5: %5.6f, NSE>0: %5.6f"%(name[0],nse_percent_5,nse_percent_0))
    def run_model_in_testing_event(self):
        saved_simulation_path = Path(self.saved_objects_dir) / "saved_normalized_testing_windows"
        list_of_simulations = os.listdir(saved_simulation_path)
        for event in list_of_simulations:
            self.one_event_run_in_model(saved_simulation_path,event)

    def run_model_in_validation_event(self, event_index=5):

        val_event = self.validation_simulations[event_index]
        normalizer = self.normalizer
        steps_behind = self.steps_behind

        sim_in_window_list = val_event.get_simulation_in_one_window(steps_behind)

        norm_sim_in_window = [normalizer.normalize_window(sim_in_window) for sim_in_window in sim_in_window_list]

        val_event_loader= DataLoader(norm_sim_in_window,batch_size=self.batch_size,num_workers=8,pin_memory=True,shuffle=True
                                     , drop_last=False)

        yhat_list =[]
        for val_list in val_event_loader:
            yhat = self.model(val_list.to(self.device))
            yhat_list.append(yhat.reshape(len(self.normalized_training_windows[0][0]['name_nodes']),-1))

        y_heads = torch.cat(yhat_list,dim=1)

        target_heads = torch.cat([real_y["norm_h_y"]for real_y in norm_sim_in_window],dim=1)[:,:y_heads.size(-1)]

        self.swmm_heads_pd = normalizer.get_unnormalized_heads_pd(target_heads)
        self.predicted_heads_pd = normalizer.get_unnormalized_heads_pd(y_heads)

        self.runoff = val_event.runoff_raw_data.iloc[self.steps_ahead_validation:self.steps_ahead_validation+y_heads.size(-1),:]


    def calculate_metrics_for_all_validation_events(self):
        metrics = {}
        for i in range(len(self.validation_simulations)):
            self.run_model_in_validation_event(i)
            metrics[i] = {
                "Heads": self.get_performance_in_heads(),
                "Flows": self.get_performance_in_flows(),
            }
        return metrics
