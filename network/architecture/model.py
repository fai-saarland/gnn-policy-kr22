import torch
import torch.nn as nn
import pytorch_lightning as pl

from architecture import AddModelBase, MaxModelBase, MaxReadoutModelBase, AddMaxModelBase, PlanFormerModelBase
from architecture import supervised_optimal_loss, selfsupervised_optimal_loss, selfsupervised_suboptimal_loss, selfsupervised_suboptimal2_loss, unsupervised_optimal_loss, unsupervised_suboptimal_loss, l1_regularization
from architecture.attention_base import AttentionModelBase
from generators.plan import policy_search

from architecture import selfsupervised_suboptimal_loss_no_solvable_labels

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
            self.log('validation_loss', validation)

    return Model

def _create_unsupervised_planformer_model_class(base: pl.LightningModule, loss):
    """Create a model class for unsupervised training that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            print("\n")
            print("learning rate")
            print(self.learning_rate)
            # TODO: HOW TO CHOOSE PATIENCE?
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
            return total

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            validation = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('validation_loss', validation)

    return Model

def _create_unsupervised_retrain_model_class(base: pl.LightningModule, loss):
    """Create a model class for unsupervised training that inherits from 'base' and uses 'loss' for training and validation."""
    class Model(base):
        def __init__(self, predicates: list, hidden_size: int, iterations: int, learning_rate: float, l1_factor: float, weight_decay: float, **kwargs):
            super().__init__(predicates, hidden_size, iterations)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay
            self.bug_states = []
            self.bug_counts = np.array([])
            self.training_step_counter = 0
            self.loss = loss
            self.bug_loss_weight = 0.1  # 0.001?
            self.min_train_loss = 0
            self.max_train_loss = 0

        def configure_optimizers(self):
            return _create_optimizer(self, self.learning_rate, self.weight_decay)

        def set_oracle(self, oracle):
            self.oracle = oracle

        def set_checkpoint_path(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def state_to_string(self, state):
            state_string = ""
            #print("\n")
            #print("STATE")
            #print(state)
            #print(state[1])
            #assert True == False
            for predicate in state[1][0].keys():  # only need to look at the first state since the successors are fixed
                state_string += f'{predicate}: {state[1][0][predicate]}\n'
            return state_string


        def get_bug_states(self):
            new_bug_states = self.oracle.get_bug_states()

            for new_bug in new_bug_states:
                bug_string = self.state_to_string(new_bug)
                if bug_string not in self.bug_dict:
                    self.bug_states.append(new_bug)
                    self.bug_dict[bug_string] = len(self.bug_states) - 1
                    self.bug_counts = np.append(self.bug_counts, 1.0)
                else:
                    bug_index = self.bug_dict[bug_string]
                    old_bug_label = self.bug_states[bug_index][0]
                    new_bug_label = new_bug[0]
                    if new_bug_label < old_bug_label:
                        self.bug_states[bug_index] = new_bug
                        self.bug_counts[bug_index] = 0.0

            #self.bug_states.extend(new_bug_states)
            #self.bug_counts = np.array([0.0 for _ in range(len(new_bug_states))])
            self.bug_counts += 1.0
            # self.bug_counts = np.append(self.bug_counts, [1.0 for _ in range(len(new_bug_states))], 0)
            # print(self.bug_counts)

            assert len(self.bug_states) == len(self.bug_counts)

        def set_update_interval(self, update_interval):
            self.update_interval = update_interval

        # TODO: ADD COLLECTION OF BUG STATES; BUG STATES COUNTER, USE SEPARATE LOSS FOR BUGS?

        # TODO: build bug batch here, also store average train loss
        def training_step(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            #print(type(train_batch))
            output = self(collated_states_with_object_counts)
            train_loss = self.loss(output, labels, solvable_labels, state_counts, self.device)
            with torch.no_grad():
                if train_loss > self.max_train_loss:
                    self.max_train_loss = train_loss
                elif train_loss < self.min_train_loss:
                    self.min_train_loss = train_loss
            #self.log('train_loss', train_loss)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train_loss + l1
            self.log('train_loss', total, prog_bar=True, on_step=False, on_epoch=True)
            #print(f'train_loss: {total}')

            # define distribution over bug states counts
            # bug_scores = 1 / np.sqrt(self.bug_counts + 1.0)
            bug_scores = 1 / self.bug_counts
            bug_probs = bug_scores / np.sum(bug_scores)
            bug_ids = np.arange(len(self.bug_states))  # TODO: COMPUTE THIS WHEN NEW BUGS ARE ADDED


            # TODO: CHECK THIS
            #print(f'bug_counts: {len(self.bug_counts)}')
            #print(f'bug_probs: {len(bug_probs)}')
            #print(f'bug_ids: {len(bug_ids)}')

            bug_batch = []
            for _ in range(len(train_batch)):
            #for _ in range(1):
                bug_id = np.random.choice(bug_ids, p=bug_probs)
                bug_batch.append(self.bug_states[bug_id])
                #print(self.bug_states[bug_id])
                #print(len(self.bug_states[bug_id]))
                # self.bug_counts[bug_id] += 1

            # print("\n")
            # print("BUG BATCH")
            # print(bug_batch)
            #print(len(bug_batch))
            #print("\n")
            #print(bug_batch[0])
            #print(len(bug_batch[0]))

            labels, collated_states_with_object_counts, solvable_labels, state_counts = self.oracle.collate(bug_batch)

            #print("\n")
            # print("COLLATED BUG BATCH")
            #print(collated_states_with_object_counts)
            #print(len(collated_states_with_object_counts))
            # print("\n")
            # print(self.oracle.collate(bug_batch))
            # print("\n")
            # print(labels)
            # print(solvable_labels)



            # output, _ = self(collated_states_with_object_counts)  # not interested in solvable prediction
            output = self(collated_states_with_object_counts)
            # bug_loss = self.loss(output, labels, solvable_labels, state_counts, self.device)  # TODO: USE L2 LOSS
            # TODO: DOES NOT PREDICTING THE SOLVABLE LABELS WITH THE BUG LOSS DETERIORATE PERFORMANCE?
            bug_loss = selfsupervised_suboptimal_loss_no_solvable_labels(output, labels, state_counts, self.device)
            # penalize output being larger than labels
            # bug_loss = torch.mean(torch.max(torch.zeros_like(output), output - labels))
            # bug_loss = torch.mean(torch.abs(torch.sub(labels, output)))  # TODO USE MSE LOSS?
            # compute mean squared error
            # bug_loss = torch.mean(torch.pow(torch.sub(labels, output), 2))

            self.log('bug_loss', bug_loss, prog_bar=True, on_step=False, on_epoch=True)
            #print(f'bug_loss: {bug_loss}')


            self.train_losses.append(total)
            self.bug_losses.append(bug_loss)

            loss = train_loss + self.bug_loss_weight * bug_loss
            assert True == False
            return loss

        def training_step_new(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            train_batch_ids = np.arange(len(labels))
            bug_ids = np.arange(len(self.bug_states))
            bug_scores = 1 / self.bug_counts
            bug_probs = bug_scores / np.sum(bug_scores)

            train_samples = []
            bug_samples = []
            for i in range(64):  # TODO: STORE BATCH SIZE
                # probabilistically decide whether to sample a train or bug state
                if np.random.choice([0, 1], p=[self.bug_loss_weight, 1-self.bug_loss_weight]) == 1:
                    # sample a train state
                    train_id = np.random.choice(train_batch_ids)
                    train_state = collated_states_with_object_counts[train_id]
                    train_label = labels[train_id]
                    train_solvable_label = solvable_labels[train_id]
                    train_state_count = state_counts[train_id]
                    train_samples.append((train_label, train_state, train_solvable_label, train_state_count))
                    #output = self(train_state)
                    #train_loss = self.loss(output, train_label, train_solvable_label, train_state_count, self.device)
                else:
                    bug_id = np.random.choice(bug_ids, p=bug_probs)
                    bug_samples.append(self.bug_states[bug_id])

            # stack train samples along each dimension of the tuple
            # TODO: IS THIS CORRECT?
            train_labels = torch.stack([sample[0] for sample in train_samples])
            train_states = torch.stack([sample[1] for sample in train_samples])
            train_solvable_labels = torch.stack([sample[2] for sample in train_samples])
            train_state_counts = torch.stack([sample[3] for sample in train_samples])

            output = self(train_states)
            train_loss = self.loss(output, train_labels, train_solvable_labels, train_state_counts, self.device)
            with torch.no_grad():
                if train_loss > self.max_train_loss:
                    self.max_train_loss = train_loss
                elif train_loss < self.min_train_loss:
                    self.min_train_loss = train_loss
            #self.log('train_loss', train_loss)
            l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train_loss + l1
            self.log('train_loss', total, prog_bar=True, on_step=False, on_epoch=True)
            #print(f'train_loss: {total}')

            labels, collated_states_with_object_counts, solvable_labels, state_counts = self.oracle.collate(bug_samples)

            # output, _ = self(collated_states_with_object_counts)  # not interested in solvable prediction
            output = self(collated_states_with_object_counts)
            bug_loss = selfsupervised_suboptimal_loss_no_solvable_labels(output, labels, state_counts, self.device)

            self.log('bug_loss', bug_loss, prog_bar=True, on_step=False, on_epoch=True)
            #print(f'bug_loss: {bug_loss}')

            self.train_losses.append(total)
            self.all_train_losses.append(total)
            self.bug_losses.append(bug_loss)
            self.all_bug_losses.append(bug_loss)

            loss = train_loss + bug_loss

            return loss

        def on_train_start(self):
            self.train_losses = []
            self.bug_losses = []
            self.bug_dict = {}
            self.all_train_losses = []
            self.all_bug_losses = []
            self.all_val_losses = []

        def on_train_epoch_end(self):
            with torch.no_grad():
                train_loss = sum(l.mean() for l in self.train_losses) / len(self.train_losses)
                print(f'epoch train loss: {train_loss}')
                print(f'min train loss: {self.min_train_loss}')
                print(f'max train loss: {self.max_train_loss}')

                val_loss = sum(l.mean() for l in self.all_val_losses) / len(self.all_val_losses)

                m = 1.0 / (self.max_train_loss - self.min_train_loss)
                b = -self.min_train_loss / (self.max_train_loss - self.min_train_loss)
                interpolated = m * train_loss + b
                self.bug_loss_weight = 1.0 - interpolated
                print(f'bug loss weight: {self.bug_loss_weight}')
                bug_loss = sum(l.mean() for l in self.bug_losses) / len(self.bug_losses)
                print(f'epoch bug loss: {bug_loss}')

                self.all_train_losses.append(train_loss.item())
                self.all_bug_losses.append(bug_loss.item())
                self.train_losses.clear()
                self.bug_losses.clear()

        def on_train_epoch_start(self):
            self.get_bug_states()

        def on_train_end(self):
            # TODO: USE CHECKPOINT PATH
            with open(self.checkpoint_path + "losses.train", "w") as f:
                f.write(json.dumps(self.all_train_losses))
            with open(self.checkpoint_path + "losses.bugs", "w") as f:
                f.write(json.dumps(self.all_bug_losses))
            with open(self.checkpoint_path + "losses.val", "w") as f:
                f.write(json.dumps(self.all_val_losses))
            # TODO: SAVE FURTHER INFO? STORE BUGS?

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            val_loss = loss(output, labels, solvable_labels, state_counts, self.device)

            #bug_ids = np.arange(len(self.bug_states))
            #bug_batch = []
            #for _ in range(len(validation_batch)):
            #    bug_id = np.random.choice(bug_ids)
            #    bug_batch.append(self.bug_states[bug_id])

            #labels, collated_states_with_object_counts, solvable_labels, state_counts = self.oracle.collate(bug_batch)

            #output = self(collated_states_with_object_counts)
            #bug_loss = selfsupervised_suboptimal_loss_no_solvable_labels(output, labels, state_counts, self.device)
            #self.log('validation_bug_loss', bug_loss, prog_bar=True, on_step=False, on_epoch=True)

            validation = val_loss #+ self.bug_loss_weight * bug_loss

            self.all_val_losses.append(validation)

            self.log('validation_loss', validation)

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

PlanFormer = _create_unsupervised_planformer_model_class(PlanFormerModelBase, selfsupervised_suboptimal_loss)

RetrainSelfsupervisedSuboptimalAddModel = _create_unsupervised_retrain_model_class(AddModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalMaxModel = _create_unsupervised_retrain_model_class(MaxModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalAddMaxModel = _create_unsupervised_retrain_model_class(AddMaxModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalMaxReadoutModel = _create_unsupervised_retrain_model_class(MaxReadoutModelBase, selfsupervised_suboptimal_loss)
RetrainSelfsupervisedSuboptimalAttentionModel = _create_unsupervised_retrain_model_class(AttentionModelBase, selfsupervised_suboptimal_loss)


