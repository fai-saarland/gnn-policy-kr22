import pytorch_lightning as pl
import torch
from typing import List
import json


def mse_loss(self, predicted, target):
    return torch.nn.MSELoss()(predicted, target)

def create_GNN(base: pl.LightningModule, loss):
    class GNN(base):
        def __init__(self, hidden_sizes: List[int], layers: int, dropout: int, learning_rate: float, l1_factor: float,
                     weight_decay: float, checkpoint_path: str, **kwargs):
            super().__init__(hidden_sizes=hidden_sizes, layers=layers, dropout=dropout, **kwargs)
            self.save_hyperparameters('learning_rate', 'l1_factor', 'weight_decay')
            self.learning_rate = learning_rate
            self.l1_factor = l1_factor
            self.weight_decay = weight_decay
            self.checkpoint_path = checkpoint_path

            self.train_losses = []
            self.validation_losses = []

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate or self.lr),
                                         weight_decay=self.weight_decay)
            # print("\n")
            # print("learning rate")
            # print(self.learning_rate)
            # print(self.lr)
            # TODO: ADD SCHEDULER!
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

            optimize = {
                'optimizer': optimizer,
                # 'lr_scheduler': scheduler,
                # 'monitor': "validation_loss",
            }
            return optimize

        def training_step(self, train_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch
            output = self(collated_states_with_object_counts)
            train = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('train_loss', train, prog_bar=True, on_step=True, on_epoch=True)
            # l1 = l1_regularization(self, self.l1_factor)
            self.log('l1_loss', l1)
            total = train + l1
            self.log('total_loss', total)
            self.train_losses.append(total)
            return total

        def validation_step(self, validation_batch, batch_index):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = validation_batch
            output = self(collated_states_with_object_counts)
            validation_loss = loss(output, labels, solvable_labels, state_counts, self.device)
            self.log('validation_loss', validation_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.validation_losses.append(validation_loss)

        #def on_train_epoch_end(self):
        #    with torch.no_grad():
        #        # compute average loss on training samples during the last epoch
        #        train_loss = sum(l.mean() for l in self.train_losses) / len(self.train_losses)
        #        print(f'epoch train loss: {train_loss}')
        #        self.train_losses.clear()
        #
        #        # compute average loss on validation samples during the last epoch
        #        validation_loss = sum(l.mean() for l in self.validation_losses) / len(self.validation_losses)
        #        print(f'epoch validation loss: {validation_loss}')
        #        self.validation_losses.clear()

        # store information about training, and validation losses
        def on_train_end(self):
            with open(self.checkpoint_path + "losses.train", "w") as f:
                f.write(json.dumps(self.all_train_losses))
            with open(self.checkpoint_path + "losses.val", "w") as f:
                f.write(json.dumps(self.all_val_losses))

    return GNN

class GraphConvolutionNetwork(pl.LightningModule):
    def __init__(self, hidden_sizes: List[int], layers: int, dropout: int, **kwargs):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.layers = layers
        self.dropout = dropout
        self.model = None