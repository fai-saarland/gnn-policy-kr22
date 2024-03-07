import pytorch_lightning as pl
import torch
from typing import List
import json


def mse_loss(predicted, target):
    target = target.view(-1, 1)
    return torch.nn.MSELoss()(predicted, target)

def create_GNN(base: pl.LightningModule, pool, loss):
    class GNN(base):
        def __init__(self, hidden_sizes: List[int], dropout: int, learning_rate: float, heads: int, **kwargs):
            super().__init__(hidden_sizes=hidden_sizes, dropout=dropout, pool=pool, heads=heads, **kwargs)
            self.save_hyperparameters('hidden_sizes', 'dropout', 'learning_rate')
            self.learning_rate = learning_rate

            self.train_losses = []
            self.validation_losses = []

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate or self.lr))#weight_decay=self.weight_decay)
            # print("\n")
            # print("learning rate")
            # print(self.learning_rate)
            # print(self.lr)
            # TODO: ADD SCHEDULER!
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

            optimize = {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': "validation_loss",
            }
            return optimize

        def training_step(self, train_batch, batch_index):
            assert self.training == True
            out = self(train_batch)
            train_loss = loss(out, train_batch.y)
            self.log('train_loss', train_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=train_batch.num_graphs)
            # l1 = l1_regularization(self, self.l1_factor)
            # self.log('l1_loss', l1)
            # total = train + l1
            # self.log('total_loss', total)
            self.train_losses.append(train_loss)
            return train_loss

        def validation_step(self, validation_batch, batch_index):
            self.training = False
            out = self(validation_batch)
            validation_loss = loss(out, validation_batch.y)
            self.log('validation_loss', validation_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.validation_losses.append(validation_loss)
            self.training = True

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
        #def on_train_end(self):
        #    with open(self.checkpoint_path + "losses.train", "w") as f:
        #        f.write(json.dumps(self.all_train_losses))
        #    with open(self.checkpoint_path + "losses.val", "w") as f:
        #        f.write(json.dumps(self.all_val_losses))

    return GNN

from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F
class GraphConvolutionNetwork(pl.LightningModule):
    def __init__(self, hidden_sizes: List[int], dropout: int, heads: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(GCNConv(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            self.norms.append(BatchNorm(self.hidden_sizes[i+1]))
        self.out = torch.nn.Linear(hidden_sizes[-1], 1)
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x

from torch_geometric.nn import GATConv
class GraphAttentionNetwork(pl.LightningModule):
    def __init__(self, hidden_sizes: List[int], dropout: int, heads: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.heads = heads
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(GATConv(self.hidden_sizes[i], self.hidden_sizes[i+1], heads=self.heads))#, concat=False))
            self.norms.append(BatchNorm(self.hidden_sizes[i+1]))
        self.out = torch.nn.Linear(hidden_sizes[-1], 1)
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x
