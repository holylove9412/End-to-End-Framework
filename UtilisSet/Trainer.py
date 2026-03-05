from torch_geometric.utils import to_dense_adj

from UtilisSet.utils import NSE_Calcu,calculate_nse_per_node
import wandb
import torch.nn.functional as F
import datetime
from abc import ABC, abstractmethod
import os

def TrainerFactory(trainer_name):
    available_trainers = {
        "Trainer_Heads": Trainer_Heads,
    }
    return available_trainers[trainer_name]


class Trainer(ABC):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        report_freq,
        switch_epoch,
        min_expected_loss,
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler
        self.report_freq = report_freq
        self.switch_epoch = switch_epoch
        self.normalizer = kwargs['normalizer']
        self.train_monitores_nodes = []
        self.trained_weights_path=kwargs['trained_weights_path']

        self.train_losses = []
        self.val_losses = []


        self.min_val_loss = 1e6
        self.best_model_parameters = copy.deepcopy(self.model.state_dict())
        self.last_model_parameters = copy.deepcopy(self.model.state_dict())
        self.epoch_best_model = -1

        self.last_loss = 100
        self.patience = 8
        self.trigger_times = 0

        self.num_nodes = kwargs['num_nodes']

        self.monitored_nodes = torch.tensor(kwargs["monitored_nodes"])

        self.min_expected_loss = min_expected_loss

        self.epoch = 1

    def train(self, train_loaders, val_loader, epochs):
        self._set_initial_state_at_first_epoch(epochs)

        try:
            self._do_training_loop(train_loaders, val_loader, epochs)
        except KeyboardInterrupt:
            pass
        else:
            self._set_final_state_when_finishing(epochs)

    def _set_initial_state_at_first_epoch(self, epochs):
        if self.epoch == 1:
            wandb.watch(self.model, self.criterion, log="all", log_freq=10)
            self._print_initial_message(epochs)
            self.total_time = 0
            self.parameters_before = copy.deepcopy(list(self.model.parameters()))
            self.start_time = time.time()

    def _set_final_state_when_finishing(self, epochs):

        self.total_time = time.time() - self.start_time
        if epochs != 0:
            self.time_per_epoch_sec = self.total_time / epochs
        else:
            self.time_per_epoch_sec = 0
        self._log_wandb_end_values()
        self._print_end_message()

    def get_history(self):
        history = {}
        history["Training loss"] = self.train_losses
        history["Validation loss"] = self.val_losses
        return history
    def get_alpha(self,N1,N2,alpha_boundary_max,alpha_spatial_max,alpha_temporal_max):
        if self.epoch < N1:
            return {'boundary': 0.0, 'spatial': 0.0, 'temporal': 0.0}
        elif self.epoch < N2:
            # 线性拉升因子
            # factor = (self.epoch - N1) / (N2 - N1)
            factor = 0.05333333333
            return {
                'boundary': alpha_boundary_max * factor,
                'spatial': alpha_spatial_max * factor,
                'temporal': alpha_temporal_max * factor,
            }
        else:

            return {
                'boundary': alpha_boundary_max,
                'spatial': alpha_spatial_max,
                'temporal': alpha_temporal_max,
            }
    def set_epoch(self, epoch):
        self.epoch = epoch

    def _load_best_parameters_in_model(self):
        return self.model.load_state_dict(self.best_model_parameters)

    def _load_last_parameters_in_model(self):
        return self.model.load_state_dict(self.last_model_parameters)

    # def _data_mask(self):
    #     self.monitored_nodes=torch.tensor([54,187,104,202,335,298])
    #     self.train_monitores_nodes = self.monitored_nodes[torch.randperm(len(self.monitored_nodes))[:4]]
        # return train_monitores_nodes
    def _do_training_loop(self, train_loaders, val_loader, epochs):

        early_stopper = EarlyStoppingController(patience=50, verbose=True,save_path=self.trained_weights_path)
        N1, N2 = 0, 500
        alpha_boundary_max = 1.0
        alpha_spatial_max = 0.5
        alpha_temporal_max = 0.2

        while self.epoch <= epochs:
            # if self.epoch <= self.switch_epoch:
            #     train_loader = train_loaders[0]
            # else:
            train_loader = train_loaders[1]
            # if (self.epoch-1)%5== 0:
            #     self._data_mask()

            self.alpha_dict = self.get_alpha(N1, N2, alpha_boundary_max,alpha_spatial_max, alpha_temporal_max)
            train_loss,train_nse,train_nse_max= self._get_train_loss(train_loader)
            val_loss,val_nse,val_nse_max = self._get_validation_loss(val_loader)

            self.scheduler.step()

            self.last_model_parameters = copy.deepcopy(self.model.state_dict())
            self._record_best_model(1-val_nse)

            self.total_time = time.time() - self.start_time
            self.time_per_epoch_sec = self.total_time / self.epoch
            remaining_time = (epochs - self.epoch) * self.time_per_epoch_sec
            current_lr = self.optimizer.param_groups[0]['lr']
            self._printCurrentStatus(epochs, train_loss, train_nse, train_nse_max,val_loss, val_nse,val_nse_max,current_lr,remaining_time)
            wandb.log(
                {
                    "Training loss": train_loss,
                    "Training nse": train_nse,
                    "Validation loss": val_loss,
                    "Validation nse": val_nse,
                    "epoch": self.epoch,
                }
            )
            early_stopper.step(float(1-val_nse), self.epoch, model=self.model)
            # if counter>20:
            #     self._data_mask()
            # if early_stopper.step(float(val_loss), self.epoch, model=self.model):
            #     break
            # if self._should_early_stop(val_loss):
            #     break
            if self.epoch == 2:
                self._check_parameters_changed_with_training()

            self.epoch += 1
            # early_stopper.restore_best_weights(self.model)

    def _check_parameters_changed_with_training(self):
        parameters_after = copy.deepcopy(list(self.model.parameters()))
        # for i in range(len(self.parameters_before)):
        #     if self._all_entries_in_two_tensors_are_close(
        #         self.parameters_before[i], parameters_after[i]
        #     ):
        #         print("Parameter is not changing: ", i)
                # warnings.warn(f"Parameter {i} is not changing")

        for i, (before, after) in enumerate(zip(self.parameters_before, parameters_after)):
            if self._all_entries_in_two_tensors_are_close(before, after):
                name = list(self.model.state_dict().keys())[i]
                print(f"参数未更新: {name}")
    def _record_best_model(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.epoch_best_model = self.epoch
            self.best_model_parameters = copy.deepcopy(self.model.state_dict())

    @abstractmethod
    def _get_train_loss(self, loader):
        pass

    @abstractmethod
    def _get_validation_loss(self, loader):
        pass

    def _should_early_stop(self, val_loss):
        stop_run = False
        if self._is_worth_stopping(val_loss):
            self.trigger_times += 1
            if self.trigger_times >= self.patience:
                print("Early stopping!", "The Current Loss:", val_loss)
                stop_run = True
        else:
            self.trigger_times = 0

        self.last_loss = val_loss
        return stop_run

    def _is_worth_stopping(self, current_loss):
        is_loss_stuck = abs(current_loss - self.last_loss) < 1e-4

        is_loss_increasing = current_loss > self.last_loss

        is_loss_exploding = current_loss > 10e5

        is_loss_higher_than_expected = current_loss > self.min_expected_loss

        return (
            is_loss_stuck
            or is_loss_exploding
            or is_loss_increasing
            or is_loss_higher_than_expected
        )

    def _print_initial_message(self, epochs):
        print(
            "train() called:model=%s, trainer=%s, opt=%s(lr=%f), epochs=%d,device=%s\n"
            % (
                type(self.model).__name__,
                type(self).__name__,
                type(self.optimizer).__name__,
                self.optimizer.param_groups[0]["lr"],
                epochs,
                self.device,
            )
        )

    def _printCurrentStatus(self, epochs, train_loss, train_nse, train_nse_max,val_loss, val_nse,val_nse_max,current_lr,remaining_time):
        epoch = self.epoch
        remaining_time_formatted = str(
            datetime.timedelta(seconds=round(remaining_time))
        )
        # if self._is_a_printing_epoch(epochs, epoch):
        print(
            "Epoch %3d/%3d, train loss: %5.6f, train nse: %5.6f,train nse_max: %5.6f,val loss: %5.6f, val nse: %5.6f, val nse_max: %5.6f,lr: %5.6f,ETA: "
            % (epoch, epochs, train_loss, train_nse,train_nse_max, val_loss, val_nse,val_nse_max,current_lr)
            + remaining_time_formatted
        )

    def _is_a_printing_epoch(self, epochs, epoch):
        return epoch == 1 or epoch % self.report_freq == 0 or epoch == epochs

    def _print_end_message(self):
        print()
        print("Best model found at epoch: ", self.epoch_best_model)
        print("Best validation loss found: %5.4f" % (self.min_val_loss))
        print()
        print("Time total:     %5.2f sec" % (self.total_time))
        print("Time per epoch: %5.2f sec" % (self.time_per_epoch_sec))

    def _log_wandb_end_values(self):
        wandb.log({"min_val_loss": self.min_val_loss})
        wandb.log({"Training time (s)": self.total_time})
        wandb.log({"Training time per epoch (s)": self.time_per_epoch_sec})

    def _all_entries_in_two_tensors_are_close(self, tensor_a, tensor_b):
        return torch.isclose(tensor_a, tensor_b).all().item()


class Trainer_Heads(Trainer):
    def _move_data_to_device(self,data, device):
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device, non_blocking=True)
        return data
    def _get_cacluloss_data(self,data):
        self.monitored_nodes = torch.tensor([0,54,187,104,202,219,335,298])
        caclu_nodes = ~torch.isin(self.monitored_nodes, self.train_monitores_nodes)
        caclu_data = data.reshape(-1, self.num_nodes, data.size(-1))
        caclu_data = caclu_data[:, self.monitored_nodes, :]
        return caclu_data.reshape(-1, data.size(-1))
        # caclu_data = caclu_data[:, monitored_nodes[caclu_nodes], :]
        # return caclu_data.reshape(-1, data.size(-1))

    def _get_mask_input(self,data):
        mask_monitored_data = torch.full_like(data, fill_value=0).reshape(-1, self.num_nodes, data.shape[-1])
        mask_monitored_data[:, self.train_monitores_nodes, :] = 1
        mask_monitored = mask_monitored_data.reshape(-1, data.shape[-1])
        return data*mask_monitored
    def calcu_spatial_loss(self,y_pred, adj_matrix,batch):
        y_pred_new = y_pred.view(batch,-1,y_pred.size(-1))
        y_pred_new = y_pred_new.permute(0,2,1).contiguous()
        y_pred_new = y_pred_new.view(batch*y_pred.size(-1),-1)

        i, j = adj_matrix.nonzero(as_tuple=True)

        nan_mask = torch.isnan(y_pred_new)


        i_nan = nan_mask[:, i]
        j_nan = nan_mask[:, j]


        valid_mask = ~(i_nan | j_nan)


        valid_i = i.unsqueeze(0).expand(y_pred_new.shape[0], -1)
        valid_j = j.unsqueeze(0).expand(y_pred_new.shape[0], -1)


        pred_i = y_pred_new.gather(1, valid_i)
        pred_j = y_pred_new.gather(1, valid_j)


        pred_i = pred_i[valid_mask]
        pred_j = pred_j[valid_mask]


        spatial_loss = torch.mean((pred_i - pred_j) ** 2)

        return spatial_loss
    def calcu_temporal_loss(self,y_pred,batch):
        y_pred_new = y_pred.view(batch, -1, y_pred.size(-1))
        y_pred_new = y_pred_new.permute(0, 2, 1).contiguous()

        temporal_loss = torch.tensor(0.0)
        if y_pred_new.dim() == 3:
            temporal_loss = torch.mean((y_pred_new[:, 1:, :] - y_pred_new[:, :-1, :]) ** 2)
        return temporal_loss

    import torch

    def nse_success_loss(self,pred, target,batch, threshold=0.5, eps=1e-3):
        pred = pred.view(batch,-1,pred.size(-1))
        target = target.view(batch, -1, target.size(-1))
        mask = ~torch.isnan(pred) & ~torch.isnan(target)
        pred = torch.where(mask, pred, torch.zeros_like(pred))
        target = torch.where(mask, target, torch.zeros_like(target))

        valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)

        mean_target = target.sum(dim=-1, keepdim=True) / valid_count

        numerator = ((target - pred) ** 2) * mask
        sum_numerator = numerator.sum(dim=-1)

        denominator = ((target - mean_target) ** 2) * mask
        sum_denominator = denominator.sum(dim=-1).clamp(min=eps)

        nse_per_node = 1 - (sum_numerator / sum_denominator)

        success = (nse_per_node > threshold).float()

        valid_nodes = (~torch.isnan(nse_per_node)).float()

        success_ratio_per_batch = (success.sum(dim=1) / valid_nodes.sum(dim=1).clamp(min=1))  # (batch,)

        nse_success_loss = 1.0 - success_ratio_per_batch.mean()

        return nse_success_loss

    def nse_loss(self,pred, target, eps=1e-3):
        mask = ~torch.isnan(target) & ~torch.isnan(pred)
        pred = torch.where(mask, pred, torch.zeros_like(pred))
        target = torch.where(mask, target, torch.zeros_like(target))

        valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_target = target.sum(dim=-1, keepdim=True) / valid_count

        numerator = ((target - pred) ** 2) * mask
        sum_numerator = numerator.sum(dim=-1)

        denominator = ((target - mean_target) ** 2) * mask
        sum_denominator = denominator.sum(dim=-1).clamp(min=eps)

        loss = sum_numerator / sum_denominator  # (batch, node)
        return loss.mean()

    def calclu_physics_loss(self, y_head, z_min, z_max):
        y_pred = y_head-z_min
        max_depth = z_max-z_min
        below_invert = torch.relu(-y_pred)
        above_ground = torch.relu(y_pred-max_depth)*(max_depth>0).float()
        phy_loss = (below_invert**2)+(above_ground**2).mean()

        return phy_loss
    def calclu_mse_loss(self,y_pred, y_true):
        mse_loss = self.criterion(y_pred,y_true)

        return mse_loss

    def _data_mask(self):

        self.train_monitores_nodes = self.monitored_nodes[torch.randperm(len(self.monitored_nodes))[:int(len(self.monitored_nodes)*0.5)]]


    def _get_train_loss(self, loader):

        self.model.train()
        self.model.to(self.device)

        window_sample = loader.dataset[0]
        adj = to_dense_adj(window_sample.edge_index, max_num_nodes=window_sample.num_nodes).squeeze(0).to(self.device)

        train_one_epoch_loss=[]
        train_one_epoch_nse = []
        train_one_epoch_nse_max = []
        for batch in loader:
            self._data_mask()
            batch['x'] = self._get_mask_input(batch.x)
            batch['node_attr'] = torch.cat([self._get_mask_input(batch.node_attr[:,:72]),batch.node_attr[:,-2:]],dim=1)

            x = self._move_data_to_device(batch, self.device)
            true_y = batch.y
            pred_y = self.model(x)[:,-12:]


            pred_y = pred_y.view(-1, self.num_nodes, 12)
            true_y = true_y.view(-1, self.num_nodes, 12)
            yhat = pred_y[:,self.monitored_nodes,:]
            y_nodes = true_y[:,self.monitored_nodes,:]
            z_min = batch.node_attr[:,1:2].view(-1, self.num_nodes, 1)[:,self.monitored_nodes,:]
            z_max = batch.node_attr[:, :1].view(-1, self.num_nodes, 1)[:, self.monitored_nodes, :]

            mse_loss = F.smooth_l1_loss(
                yhat,
                y_nodes,
                reduction='mean'
            )*x.size(0)


            physics_loss = torch.mean(self.calclu_physics_loss(
                yhat,
                z_min,
                z_max
            ))* x.size(0)

            alpha_nse = min(self.epoch / 20, 1.0) * 1
            loss = mse_loss+physics_loss

            train_nse = torch.mean(calculate_nse_per_node(
                pred_y,
                true_y,
                self.normalizer,
                x.batch_size
            ))*x.batch_size
            # if train_nse>0:
            #     print('train_nse>0')
            train_nse_max = torch.max(calculate_nse_per_node(
                pred_y,
                true_y,
                self.normalizer,
                x.batch_size
            ))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()


            train_one_epoch_loss.append(loss.item())
            train_one_epoch_nse.append(train_nse.item())
            train_one_epoch_nse_max.append(train_nse_max.item())

        # ===================== 汇总 =====================
        train_loss = sum(train_one_epoch_loss)/(len(train_one_epoch_loss)*loader.batch_size)
        train_nses = sum(train_one_epoch_nse)/ (len(train_one_epoch_nse)*loader.batch_size)

        train_nses_max = max(train_one_epoch_nse_max)


        self.train_losses.append(train_loss)

        return train_loss, train_nses,train_nses_max

    def _get_validation_loss(self, loader):
        self.model.eval()
        self.model.to(self.device)

        window_sample = loader.dataset[0]
        adj = to_dense_adj(window_sample.edge_index, max_num_nodes=window_sample.num_nodes).squeeze(0).to(self.device)
        val_one_epoch_loss=[]
        val_one_epoch_nse = []
        val_one_epoch_nse_max = []
        with torch.no_grad():
            for batch in loader:
                self._data_mask()
                batch['x'] = self._get_mask_input(batch.x)
                batch['node_attr'] = torch.cat([self._get_mask_input(batch.node_attr[:, :72]), batch.node_attr[:, -2:]],
                                               dim=1)

                x = self._move_data_to_device(batch, self.device)
                true_y = batch.y
                pred_y = self.model(x)[:,-12:]

                pred_y = pred_y.view(-1, self.num_nodes, 12)
                true_y = true_y.view(-1, self.num_nodes, 12)
                yhat = pred_y[:, self.monitored_nodes, :]
                y_nodes = true_y[:, self.monitored_nodes, :]
                z_min = batch.node_attr[:, 1:2].view(-1, self.num_nodes, 1)[:, self.monitored_nodes, :]
                z_max = batch.node_attr[:, :1].view(-1, self.num_nodes, 1)[:, self.monitored_nodes, :]

                mse_loss = F.smooth_l1_loss(
                    yhat,
                    y_nodes,
                    reduction='mean'
                ) * x.size(0)

                physics_loss = torch.mean(self.calclu_physics_loss(
                    yhat,
                    z_min,
                    z_max
                ))*x.size(0)

                alpha_nse = min(self.epoch / 10, 1.0) * 1
                loss = mse_loss+physics_loss
                val_nse_mean = torch.mean(calculate_nse_per_node(
                    pred_y,
                    true_y,
                    self.normalizer,
                    x.batch_size
                ))*x.batch_size
                val_nse_max = torch.max(calculate_nse_per_node(
                    pred_y,
                    true_y,
                    self.normalizer,
                    x.batch_size
                ))


                val_one_epoch_loss.append(loss.item())
                val_one_epoch_nse.append(val_nse_mean.item())
                val_one_epoch_nse_max.append(val_nse_max.item())


        val_loss = sum(val_one_epoch_loss) / (len(val_one_epoch_loss)*loader.batch_size)
        val_nses = sum(val_one_epoch_nse) / (len(val_one_epoch_nse)*loader.batch_size)
        val_nses_max = max(val_one_epoch_nse_max)

        self.val_losses.append(val_loss)


        return val_loss, val_nses,val_nses_max


class IncorrectTrainerException(Exception):
    pass
import copy
import time
import torch

class EarlyStoppingController:
    def __init__(self, patience=5, min_delta=1e-4, verbose=True, save_path=None, save_optimizer=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path
        self.save_optimizer = save_optimizer

        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.best_state_dict = None
        self.early_stop = False

    def step(self, current_loss, epoch, model=None, optimizer=None):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0

            if model:
                self.best_state_dict = copy.deepcopy(model.state_dict())

            if self.verbose:
                print(f"[{time.strftime('%H:%M:%S')}] New best loss: {current_loss:.6f} at epoch {epoch}")

            # 自动保存
            if self.save_path and model:
                self.save_checkpoint(
                    model=model,
                    path=self.save_path,
                    optimizer=optimizer if self.save_optimizer else None,
                    epoch=epoch,
                    loss=current_loss
                )
        else:
            self.counter += 1
            if self.verbose:
                print(f"[{time.strftime('%H:%M:%S')}] No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                print(f"[{time.strftime('%H:%M:%S')}] Early stopping triggered at epoch {epoch}")

        return self.early_stop

    def restore_best_weights(self, model):
        if self.best_state_dict is not None and self.counter>self.patience:
            self.counter=0
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"Restored model to best epoch {self.best_epoch} with loss {self.best_loss:.6f}")

    def save_checkpoint(self,model, path, optimizer=None, epoch=None, loss=None, extra=None):
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if extra:
            checkpoint.update(extra)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f" Checkpoint saved to: {path}")

