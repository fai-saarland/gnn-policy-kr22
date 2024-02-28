import torch
import torch.nn as nn
import pytorch_lightning as pl

from architecture import AddModelBase, MaxModelBase, MaxReadoutModelBase, AddMaxModelBase, PlanFormerModelBase
from architecture import supervised_optimal_loss, selfsupervised_optimal_loss, selfsupervised_suboptimal_loss, selfsupervised_suboptimal2_loss, unsupervised_optimal_loss, unsupervised_suboptimal_loss, l1_regularization
from architecture.attention_base import AttentionModelBase
from generators.plan import policy_search

from architecture import selfsupervised_suboptimal_loss_no_solvable_labels

from architecture import mean_squared_error_loss

import numpy as np

import json

_max_trace_length = 4

def set_max_trace_length(max_length: int):
    global _max_trace_length
    _max_trace_length = max_length

def _create_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def _create_supervised_model_class(base: pl.LightningModule, loss):
    """Create a model class for supervised training that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def training_step(self, train_batch, batch_index):
            states, target = train_batch
            output = self(states)
            train = loss(output, target)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            return total

        def validation_step(self, validation_batch, batch_index):
            states, target = validation_batch
            output = self(states)
            validation = loss(output, target)
            self.log('validation_loss', validation)

    return Model

def _create_unsupervised_model_class(base: pl.LightningModule, loss):
    """Create a model class for unsupervised training that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def training_step(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            output = self(collated_states_with_object_counts)
            train = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            return total

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            validation = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('validation_loss', validation, on_step=False, on_epoch=True)

    return Model

def _create_online_model_class(base: pl.LightningModule, loss):
    """Create a model class for online training (unsupervised) that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def training_step(self, train_batch, batch_index):
            train = 0.0
            for problem, encoding in train_batch:
                action_trace, state_trace, value_trace, is_solution = policy_search(problem['actions'], problem['initial'], problem['goal'], encoding, self, _max_trace_length)
                values = []
                goals = []
                counts = []
                for index in range(len(value_trace) - 1):
                    values.append(value_trace[index])
                    values.append(value_trace[index + 1])
                    goals.append(False)
                    counts.append(2)
                if is_solution:
                    values.append(value_trace[-1])
                    goals.append(True)
                    counts.append(1)
                train += loss(torch.stack(values), goals, counts, self.device)
            train /= len(train_batch)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            return total

        def validation_step(self, validation_batch, batch_index):
            validation = 0.0
            for problem, encoding in validation_batch:
                action_trace, state_trace, value_trace, is_solution = policy_search(problem['actions'], problem['initial'], problem['goal'], encoding, self, _max_trace_length)
                values = []
                goals = []
                counts = []
                for index in range(len(value_trace) - 1):
                    values.append(value_trace[index])
                    values.append(value_trace[index + 1])
                    goals.append(False)
                    counts.append(2)
                if is_solution:
                    values.append(value_trace[-1])
                    goals.append(True)
                    counts.append(1)
                validation += loss(torch.stack(values), goals, counts, self.device)
            validation /= len(validation_batch)
            self.log('validation_loss', validation)

    return Model

def _create_unsupervised_retrain_model_class(base: pl.LightningModule, loss):
    """Create a model class for retraining of models trained without supervision that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            # training hyperparameters
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay
            self.loss = loss
            self.original_validation_loss = np.inf

            # variables for bugs
            self.bug_states = []
            self.bug_counts = np.array([])
            self.bug_ids = []
            self.bug_dict = {}
            self.update_counter = 0

            # variables for validation bugs
            self.val_bug_states = []
            self.val_bug_counts = np.array([])
            self.val_bug_ids = []
            self.val_bug_dict = {}
            self.val_update_counter = 0

            # variables for computing of bug loss weight and logging
            self.bug_loss_weight = 0.1  # This is the bug loss weight in the first epoch!
            self.min_train_loss = 0
            self.max_train_loss = 0
            self.train_losses = []
            self.bug_losses = []
            self.val_losses = []
            self.val_bug_losses = []
            self.all_train_losses = []
            self.all_bug_losses = []
            self.all_val_losses = []
            self.all_val_bug_losses = []

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def initialize(self, oracle, checkpoint_path, update_interval, no_bug_loss_weight, no_bug_counts):
            self.oracle = oracle
            self.checkpoint_path = checkpoint_path
            self.update_interval = update_interval
            self.no_bug_loss_weight = no_bug_loss_weight
            self.no_bug_counts = no_bug_counts

        # map a state to a string such that we can check whether we have seen this state before
        def state_to_string(self, state):
            state_string = ""
            for predicate in state[1][0].keys():  # only need to look at the first state since the successors are fixed
                state_string += f'{predicate}: {state[1][0][predicate]} '
            return state_string

        # get new bug states from oracle, if a state is found again but with a better label we replace the old label
        def get_bug_states(self):
            new_bug_states = self.oracle.get_bug_states()

            for new_bug in new_bug_states:
                bug_string = self.state_to_string(new_bug)
                if bug_string not in self.bug_dict:
                    self.bug_states.append(new_bug)
                    self.bug_dict[bug_string] = len(self.bug_states) - 1
                    self.bug_counts = np.append(self.bug_counts, 0.0)
                else:
                    bug_index = self.bug_dict[bug_string]
                    old_bug_label = self.bug_states[bug_index][0]
                    new_bug_label = new_bug[0]
                    if new_bug_label < old_bug_label:
                        self.bug_states[bug_index] = new_bug
                        self.bug_counts[bug_index] = 0.0

            self.bug_counts += 1.0
            self.bug_ids = np.arange(len(self.bug_states))

            assert len(self.bug_states) == len(self.bug_counts)

        def get_val_bug_states(self):
            new_val_bug_states = self.oracle.get_val_bug_states()

            for new_val_bug in new_val_bug_states:
                val_bug_string = self.state_to_string(new_val_bug)
                if val_bug_string not in self.val_bug_dict:
                    self.val_bug_states.append(new_val_bug)
                    self.val_bug_dict[val_bug_string] = len(self.val_bug_states) - 1
                    self.val_bug_counts = np.append(self.val_bug_counts, 0.0)
                else:
                    val_bug_index = self.val_bug_dict[val_bug_string]
                    old_val_bug_label = self.val_bug_states[val_bug_index][0]
                    new_val_bug_label = new_val_bug[0]
                    if new_val_bug_label < old_val_bug_label:
                        self.bug_states[val_bug_index] = new_val_bug
                        self.bug_counts[val_bug_index] = 0.0

            self.val_bug_counts += 1.0
            self.val_bug_ids = np.arange(len(self.val_bug_states))

            assert len(self.val_bug_states) == len(self.val_bug_counts)

        def training_step(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            output = self(collated_states_with_object_counts)
            train_loss = self.loss(output, labels, solvable_labels, state_counts, self.device)

            # these values are used for interpolation
            with torch.no_grad():
                if train_loss > self.max_train_loss:
                    self.max_train_loss = train_loss
                elif train_loss < self.min_train_loss:
                    self.min_train_loss = train_loss


            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train_loss + l1
            self.log('train_loss', total, prog_bar=True, on_step=False, on_epoch=True)

            if len(self.bug_states) == 0:
                self.train_losses.append(total)
                return total

            else:
                # define distribution over bug states counts
                if not self.no_bug_counts:
                    bug_scores = 1 / self.bug_counts  # prioritize newer bugs
                else:
                    bug_scores = np.ones(len(self.bug_counts))  # sample uniformly

                bug_probs = bug_scores / np.sum(bug_scores)


                # sample bug states
                bug_batch = []
                for _ in range(len(train_batch)):
                    bug_id = np.random.choice(self.bug_ids, p=bug_probs)
                    bug_batch.append(self.bug_states[bug_id])

                labels, collated_states_with_object_counts, solvable_labels, state_counts = self.oracle.collate(bug_batch)

                output = self(collated_states_with_object_counts)
                # TODO: DOES NOT PREDICTING THE SOLVABLE LABELS WITH THE BUG LOSS DETERIORATE PERFORMANCE?
                # bug_loss = selfsupervised_suboptimal_loss_no_solvable_labels(output, labels, state_counts, self.device)
                bug_loss = self.loss(output, labels, solvable_labels, state_counts, self.device)
                self.log('bug_loss', bug_loss, prog_bar=True, on_step=False, on_epoch=True)

                self.train_losses.append(total)
                self.bug_losses.append(bug_loss)

                if not self.no_bug_loss_weight:
                    loss = train_loss + self.bug_loss_weight * bug_loss
                else:
                    loss = train_loss + bug_loss

                return loss

        # when we load bugs only once at the start of the training
        def on_train_start(self):
            if self.update_interval == -1:
                self.get_bug_states()
                if self.oracle.val_bugs is not None:
                    self.get_val_bug_states()

        # when we iteratively load new bugs during training
        def on_train_epoch_start(self):
            if self.update_interval != -1 and self.update_counter % self.update_interval == 0:
                self.get_bug_states()
                self.update_counter += 1

        def on_train_epoch_end(self):
            with torch.no_grad():
                # compute average loss on training samples during the last epoch
                train_loss = sum(l.mean() for l in self.train_losses) / len(self.train_losses)
                print(f'epoch train loss: {train_loss}')
                print(f'min train loss: {self.min_train_loss}')
                print(f'max train loss: {self.max_train_loss}')

                self.all_train_losses.append(train_loss.item())
                self.train_losses.clear()

                if len(self.bug_states) != 0:
                    # linearly interpolate between min and max train loss
                    m = 1.0 / (self.max_train_loss - self.min_train_loss)
                    b = -self.min_train_loss / (self.max_train_loss - self.min_train_loss)
                    interpolated = m * train_loss + b

                    # update bug loss weight
                    self.bug_loss_weight = 1.0 - interpolated
                    print(f'bug loss weight: {self.bug_loss_weight}')
                    bug_loss = sum(l.mean() for l in self.bug_losses) / len(self.bug_losses)
                    print(f'epoch bug loss: {bug_loss}')

                    self.all_bug_losses.append(bug_loss.item())
                    self.bug_losses.clear()

                if len(self.val_bug_states) != 0:
                    val_bug_loss = sum(l.mean() for l in self.val_bug_losses) / len(self.val_bug_losses)
                    print(f'epoch val bug loss: {val_bug_loss}')

                    self.all_val_bug_losses.append(val_bug_loss.item())
                    self.val_bug_losses.clear()

        # store information about training, validation, and bug losses
        def on_train_end(self):
            with open(self.checkpoint_path + "losses.train", "w") as f:
                f.write(json.dumps(self.all_train_losses))
            with open(self.checkpoint_path + "losses.bugs", "w") as f:
                f.write(json.dumps(self.all_bug_losses))
            with open(self.checkpoint_path + "losses.val", "w") as f:
                f.write(json.dumps(self.all_val_losses))

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            validation_loss = self.loss(output, labels, solvable_labels, state_counts, self.device)

            self.val_losses.append(validation_loss)

            if len(self.val_bug_states) == 0:
                self.log('validation_loss', validation_loss, prog_bar=True, on_step=False, on_epoch=True)
                return validation_loss

            else:
                val_bug_scores = np.ones(len(self.val_bug_counts))  # sample uniformly
                val_bug_probs = val_bug_scores / np.sum(val_bug_scores)

                # sample bug states
                val_bug_batch = []
                for _ in range(len(validation_batch)):
                    val_bug_id = np.random.choice(self.val_bug_ids, p=val_bug_probs)
                    val_bug_batch.append(self.val_bug_states[val_bug_id])

                labels, collated_states_with_object_counts, solvable_labels, state_counts = self.oracle.collate(val_bug_batch)

                output = self(collated_states_with_object_counts)
                val_bug_loss = self.loss(output, labels, solvable_labels, state_counts, self.device)
                self.val_bug_losses.append(val_bug_loss)

                self.log('val_bug_loss', val_bug_loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log('val_loss', validation_loss, prog_bar=True, on_step=False, on_epoch=True)

                # TODO: HOW TO DEFINE A WEIGHTING FOR VALIDATION?
                loss = validation_loss + val_bug_loss

                self.log('validation_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

                return loss

    return Model

def _create_mse_model_class(base: pl.LightningModule, loss):
    """Create a model class for supervised training using mean squared error."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def training_step(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            output = self(collated_states_with_object_counts)
            train = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            return total

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            validation = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('validation_loss', validation, on_step=False, on_epoch=True)

    return Model

def _create_unsupervised_planformer_model_class(base: pl.LightningModule, loss):
    """Create a model class for unsupervised training that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float,
                     weight_decay: float, d_model: int, n_heads: int, d_ff: int, drop: float, n_layers: int, **kwargs):
            super().__init__(predicates, hidden_size, iterations, d_model, n_heads, d_ff, drop, n_layers)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay
            self.train_losses =  []

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate or self.lr), weight_decay=self.weight_decay)
            #print("\n")
            #print("learning rate")
            #print(self.learning_rate)
            #print(self.lr)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

            optimize = {
                'optimizer': optimizer,
                #'lr_scheduler': scheduler,
                #'monitor': "validation_loss",
            }
            return optimize

        def training_step(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            output = self(collated_states_with_object_counts)
            train = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('train_loss', train)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            self.train_losses.append(total)
            return total

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            validation = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('validation_loss', validation)

        def on_train_epoch_end(self):
            with torch.no_grad():
                # compute average loss on training samples during the last epoch
                train_loss = sum(l.mean() for l in self.train_losses) / len(self.train_losses)
                print(f'epoch train loss: {train_loss}')
                self.train_losses.clear()

    return Model

PlanFormer = _create_unsupervised_planformer_model_class(PlanFormerModelBase, mean_squared_error_loss)

SupervisedOptimalAddModel = _create_supervised_model_class(AddModelBase, supervised_optimal_loss)
SupervisedOptimalMaxModel = _create_supervised_model_class(MaxModelBase, supervised_optimal_loss)
SupervisedOptimalAddMaxModel = _create_supervised_model_class(AddMaxModelBase, supervised_optimal_loss)
SupervisedOptimalMaxReadoutModel = _create_supervised_model_class(MaxReadoutModelBase, supervised_optimal_loss)
SupervisedOptimalAttentionModel = _create_supervised_model_class(AttentionModelBase, supervised_optimal_loss)

SelfsupervisedOptimalAddModel = _create_unsupervised_model_class(AddModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, selfsupervised_optimal_loss)
SelfsupervisedOptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, selfsupervised_optimal_loss)

SelfsupervisedSuboptimalAddModel = _create_unsupervised_model_class(AddModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, selfsupervised_suboptimal_loss)
SelfsupervisedSuboptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, selfsupervised_suboptimal_loss)

SelfsupervisedSuboptimalAddModel2 = _create_unsupervised_model_class(AddModelBase, selfsupervised_suboptimal2_loss)
SelfsupervisedSuboptimalMaxModel2 = _create_unsupervised_model_class(MaxModelBase, selfsupervised_suboptimal2_loss)
SelfsupervisedSuboptimalAddMaxModel2 = _create_unsupervised_model_class(AddMaxModelBase, selfsupervised_suboptimal2_loss)
SelfsupervisedSuboptimalMaxReadoutModel2 = _create_unsupervised_model_class(MaxReadoutModelBase, selfsupervised_suboptimal2_loss)

UnsupervisedOptimalAddModel = _create_unsupervised_model_class(AddModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, unsupervised_optimal_loss)
UnsupervisedOptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, unsupervised_optimal_loss)

UnsupervisedSuboptimalAddModel = _create_unsupervised_model_class(AddModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalMaxModel = _create_unsupervised_model_class(MaxModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalAddMaxModel = _create_unsupervised_model_class(AddMaxModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalMaxReadoutModel = _create_unsupervised_model_class(MaxReadoutModelBase, unsupervised_suboptimal_loss)
UnsupervisedSuboptimalAttentionModel = _create_unsupervised_model_class(AttentionModelBase, unsupervised_suboptimal_loss)

OnlineOptimalAddModel = _create_online_model_class(AddModelBase, unsupervised_optimal_loss)
OnlineOptimalMaxModel = _create_online_model_class(MaxModelBase, unsupervised_optimal_loss)
OnlineOptimalAddMaxModel = _create_online_model_class(AddMaxModelBase, unsupervised_optimal_loss)
OnlineOptimalMaxReadoutModel = _create_online_model_class(MaxReadoutModelBase, unsupervised_optimal_loss)
OnlineOptimalAttentionModel = _create_online_model_class(AttentionModelBase, unsupervised_optimal_loss)

RetrainSelfsupervisedSuboptimalAddModel = _create_unsupervised_retrain_model_class(AddModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalMaxModel = _create_unsupervised_retrain_model_class(MaxModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalAddMaxModel = _create_unsupervised_retrain_model_class(AddMaxModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalMaxReadoutModel = _create_unsupervised_retrain_model_class(MaxReadoutModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalAttentionModel = _create_unsupervised_retrain_model_class(AttentionModelBase, selfsupervised_suboptimal_loss)

MSEMaxModel = _create_mse_model_class(MaxModelBase, mean_squared_error_loss)


