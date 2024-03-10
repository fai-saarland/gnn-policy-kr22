import pytorch_lightning as pl
import torch
import numpy as np
import json


def mse_loss(predicted, target):
    target = target.view(-1, 1)
    return torch.nn.MSELoss()(predicted, target)

def create_GNN(base: pl.LightningModule, pool, loss):
    class GNN(base):
        def __init__(self, num_layers: int, hidden_size: int, dropout: int, learning_rate: float, heads: int, weight_decay: float, **kwargs):
            super().__init__(num_layers=num_layers, hidden_size=hidden_size, dropout=dropout, pool=pool, heads=heads, **kwargs)
            self.save_hyperparameters('num_layers', 'hidden_size', 'dropout', 'learning_rate', 'heads', 'weight_decay')
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

            self.train_losses = []
            self.validation_losses = []

        def set_checkpoint_path(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate or self.lr), weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)

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
            self.train_losses.append(train_loss.item())
            return train_loss

        def validation_step(self, validation_batch, batch_index):
            self.training = False
            out = self(validation_batch)
            validation_loss = loss(out, validation_batch.y)
            self.log('validation_loss', validation_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.validation_losses.append(validation_loss.item())
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
        def on_train_end(self):
            with open(self.checkpoint_path + "losses.train", "w") as f:
                f.write(json.dumps(self.train_losses))
            with open(self.checkpoint_path + "losses.val", "w") as f:
                f.write(json.dumps(self.validation_losses))

    return GNN

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphNorm
import torch.nn.functional as F
class GraphConvolutionNetwork(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [5] + [hidden_size] * (num_layers)
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(GCNConv(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            self.norms.append(GraphNorm(self.hidden_sizes[i+1]))
        self.out = torch.nn.Linear(self.hidden_sizes[-1], 1)
    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x

from torch_geometric.nn import GCN2Conv
# TODO: HOW TO CONSTRUCT THIS?
class GraphConvolutionNetworkV2(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [5] + [hidden_size] * num_layers
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(GCN2Conv(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            self.norms.append(GraphNorm(self.hidden_sizes[i+1]))
        self.out = torch.nn.Linear(self.hidden_sizes[-1], 1)
    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x

from torch_geometric.nn import GINConv
class GraphIsomorphismNetwork(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [5] + [hidden_size] * num_layers
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(GINConv(torch.nn.Sequential(torch.nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]),
                                                           torch.nn.BatchNorm1d(self.hidden_sizes[i+1]),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(self.hidden_sizes[i+1], self.hidden_sizes[i+1]),
                                                           torch.nn.ReLU())))
        lin_dim = (self.hidden_sizes[-1]*(len(self.hidden_sizes)-1)) + self.hidden_sizes[0]
        self.lin1 = torch.nn.Linear(lin_dim, lin_dim)
        self.lin2 = torch.nn.Linear(lin_dim, 1)
    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        node_embeddings = []
        node_embeddings.append(x)
        for i in range(len(self.layers)):
            prev_h = node_embeddings[-1]
            new_h = self.layers[i](prev_h, edge_index)
            new_h = F.dropout(new_h, p=self.dropout, training=self.training)
            node_embeddings.append(new_h)

        layer_readouts = [self.pool(h, batch) for h in node_embeddings]
        h = torch.cat(layer_readouts, dim=1)

        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h

from torch_geometric.nn import GATConv
class GraphAttentionNetwork(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [5] + [hidden_size] * num_layers
        self.dropout = dropout
        self.heads = heads
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        for i in range(len(self.hidden_sizes)-1):
            if i == 0:
                self.layers.append(GATConv(self.hidden_sizes[i], self.hidden_sizes[i+1], heads=self.heads))
                self.norms.append(GraphNorm(self.hidden_sizes[i + 1] * self.heads))
            elif i > 0 and i < len(self.hidden_sizes)-2:
                self.layers.append(GATConv(self.hidden_sizes[i]*self.heads, self.hidden_sizes[i+1], heads=self.heads))
                self.norms.append(GraphNorm(self.hidden_sizes[i + 1] * self.heads))
            elif i == len(self.hidden_sizes)-2:
                self.layers.append(GATConv(self.hidden_sizes[i]*self.heads, self.hidden_sizes[i+1], heads=self.heads, concat=False))
                self.norms.append(GraphNorm(self.hidden_sizes[i + 1]))
        self.out = torch.nn.Linear(self.hidden_sizes[-1], 1)
    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x

from torch_geometric.nn import GATv2Conv
class GraphAttentionNetworkV2(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [5] + [hidden_size] * num_layers
        self.dropout = dropout
        self.heads = heads
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        for i in range(len(self.hidden_sizes)-1):
            if i == 0:
                self.layers.append(GATv2Conv(self.hidden_sizes[i], self.hidden_sizes[i+1], heads=self.heads))
                self.norms.append(GraphNorm(self.hidden_sizes[i + 1] * self.heads))
            elif i > 0 and i < len(self.hidden_sizes)-2:
                self.layers.append(GATv2Conv(self.hidden_sizes[i]*self.heads, self.hidden_sizes[i+1], heads=self.heads))
                self.norms.append(GraphNorm(self.hidden_sizes[i + 1] * self.heads))
            elif i == len(self.hidden_sizes)-2:
                self.layers.append(GATv2Conv(self.hidden_sizes[i]*self.heads, self.hidden_sizes[i+1], heads=self.heads, concat=False))
                self.norms.append(GraphNorm(self.hidden_sizes[i + 1]))
        self.out = torch.nn.Linear(self.hidden_sizes[-1], 1)
    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x
