import pytorch_lightning as pl
import torch
import numpy as np
import json
from pathlib import Path
from utils_old import planning

def mse_loss(predicted, target):
    target = target.view(-1, 1)
    return torch.nn.MSELoss()(predicted, target)

def mae_loss(predicted, target):
    target = target.view(-1, 1)
    return torch.nn.L1Loss()(predicted, target)

def create_GNN(base: pl.LightningModule, pool, loss):
    class GNN(base):
        def __init__(self, num_layers: int, hidden_size: int, dropout: int, learning_rate: float, heads: int, weight_decay: float, max_arity=2, **kwargs):
            super().__init__(num_layers=num_layers, hidden_size=hidden_size, dropout=dropout, pool=pool, heads=heads, max_arity=max_arity, **kwargs)
            self.save_hyperparameters('num_layers', 'hidden_size', 'dropout', 'learning_rate', 'heads', 'max_arity', 'weight_decay')
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

            self.train_losses = []
            self.all_train_losses = []
            self.validation_losses = []
            self.all_validation_losses = []

        def set_checkpoint_path(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def enable_coverage_validation(self, validation_instances, decoded_predicate_dict, decoded_predicate_ids, max_arity, args, domain_file):
            self.validation_instances = validation_instances
            self.decoded_predicate_dict = decoded_predicate_dict
            self.decoded_predicate_ids = decoded_predicate_ids
            self.max_arity = max_arity
            self.args = args
            self.domain_file = domain_file

            self.coverage_validation = True
            self.coverages = []
            self.avg_plan_lengths = []
            self.best_coverage = 0
            self.best_avg_plan_quality = float('inf')
            self.best_policy_quality = 0.0

        def configure_optimizers(self):
            # TODO: USE ADAMW?
            # self.optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate or self.lr))
            self.optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate or self.lr), weight_decay=self.weight_decay)
            # TODO: USE COSINE SCHEDULE WITH FIXED NUMBER OF EPOCHS
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=25, verbose=True)

            optimize = {
                'optimizer': self.optimizer,
                'lr_scheduler': self.scheduler,
                'monitor': "validation_loss",
            }
            return optimize

        def training_step(self, train_batch, batch_index):
            out = self(train_batch)
            train_loss = loss(out, train_batch.y)
            self.log('train_loss', train_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=train_batch.num_graphs)
            self.train_losses.append(train_loss.item())
            return train_loss

        def validation_step(self, validation_batch, batch_index):
            out = self(validation_batch)
            validation_loss = loss(out, validation_batch.y)
            self.log('validation_loss', validation_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=validation_batch.num_graphs)
            self.validation_losses.append(validation_loss.item())

            return validation_loss

        def on_validation_epoch_end(self):
            # print("\n")
            # print("LEARNING RATE:", self.scheduler.optimizer.param_groups[0]['lr'])

            avg_train_loss = np.mean(self.train_losses)
            self.all_train_losses.append(avg_train_loss)
            self.train_losses.clear()
            avg_validation_loss = np.mean(self.validation_losses)
            self.all_validation_losses.append(avg_validation_loss)
            self.validation_losses.clear()

            if self.coverage_validation:
                solved = []
                plan_lenghts = []
                for validation_instance in self.validation_instances:
                    result_string, action_trace, is_solution = planning(self.decoded_predicate_dict, self.decoded_predicate_ids,
                                                                        self.max_arity, self.args, None, self,
                                                                        self.domain_file, validation_instance, self.device)
                    if is_solution:
                        solved.append(1)
                        plan_lenghts.append(len(action_trace))
                    else:
                        solved.append(0)

                if len(solved) == 0:
                    coverage = 0.0
                else:
                    coverage = round(sum(solved) / len(solved), 3)

                if len(plan_lenghts) == 0:
                    avg_plan_length = 10000.0
                else:
                    avg_plan_length = round(sum(plan_lenghts) / len(plan_lenghts), 3)

                # TODO: to have a perfect ranking of ALL policies we would need to store all of them and evaluate them afterward, however
                # we only care about the best one anyway
                # policy quality is incremented whenever the policy improves, allowing us to keep track of the best policies
                if coverage > self.best_coverage:
                    self.best_coverage = coverage
                    self.best_avg_plan_quality = avg_plan_length
                    self.best_policy_quality += 1.0
                    quality = self.best_policy_quality
                elif coverage == self.best_coverage and avg_plan_length < self.best_avg_plan_quality:
                    self.best_avg_plan_quality = avg_plan_length
                    self.best_policy_quality += 1.0
                    quality = self.best_policy_quality
                else:
                    quality = 0.0

                self.coverages.append(coverage)
                self.avg_plan_lengths.append(avg_plan_length)

                self.log('coverage', coverage, prog_bar=True, on_step=False, on_epoch=True)
                self.log('avg_plan_length', avg_plan_length, prog_bar=True, on_step=False, on_epoch=True)
                self.log('quality', quality, prog_bar=True, on_step=False, on_epoch=True)

        # store information about training, and validation losses
        def on_train_end(self):
            with open(self.checkpoint_path + "losses.train", "w") as f:
                f.write(json.dumps(self.all_train_losses))
            with open(self.checkpoint_path + "losses.val", "w") as f:
                f.write(json.dumps(self.all_validation_losses))

            if self.coverage_validation:
                with open(self.checkpoint_path + "losses.coverage", "w") as f:
                    f.write(json.dumps(self.coverages))
                with open(self.checkpoint_path + "losses.avg_plan_length", "w") as f:
                    f.write(json.dumps(self.avg_plan_lengths))

    return GNN

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphNorm
import torch.nn.functional as F
class GraphConvolutionNetwork(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [max_arity+3] + [hidden_size] * (num_layers)
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
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [max_arity+3] + [hidden_size] * num_layers
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
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [max_arity+3] + [hidden_size] * num_layers
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
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [max_arity+3] + [hidden_size] * num_layers
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
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int,  pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [max_arity+3] + [hidden_size] * num_layers
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

from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.utils import to_dense_batch
from torch.nn import Dropout, Sequential, Linear
from torch.nn import LayerNorm, ReLU

# TODO: THIS DOES NOT WORK!!!
class Transformer2(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [hidden_size] * num_layers
        self.dropout = dropout
        self.attention_layers = torch.nn.ModuleList()
        self.attention_norms = torch.nn.ModuleList()
        self.mlp_layers = torch.nn.ModuleList()
        self.mlp_norms = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        self.node2token = Linear(max_arity+3, self.hidden_sizes[0])
        self.input_norm = LayerNorm(self.hidden_sizes[0])

        for i in range(len(self.hidden_sizes) - 1):
            self.attention_layers.append(PerformerAttention(self.hidden_sizes[i], heads=heads))
            self.attention_norms.append(LayerNorm(self.hidden_sizes[i]))
            self.mlp_layers.append(Sequential(Linear(self.hidden_sizes[i], self.hidden_sizes[i]*2),
                                              ReLU(),
                                              Dropout(self.dropout),
                                              Linear(self.hidden_sizes[i]*2, self.hidden_sizes[i])
                                              ))
            self.mlp_norms.append(LayerNorm(self.hidden_sizes[i]))

        self.out = Sequential(Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]*2),
                              ReLU(),
                              Dropout(self.dropout),
                              Linear(self.hidden_sizes[-1]*2, 1))


    def forward(self, data):
        nodes, mask = to_dense_batch(data.x, data.batch)

        h = self.node2token(nodes)
        h = self.input_norm(h)
        for i in range(len(self.hidden_sizes) - 1):
            # Performer self-attention
            _h = h
            h = self.attention_layers[i](h, mask)
            #h = h[mask]
            # Add and norm
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.attention_norms[i](h + _h)
            # Position-wise FFN
            _h = h
            h = self.mlp_layers[i](h)
            # Add and norm
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.mlp_norms[i](h + _h)

        h = self.pool(h, data.batch)
        out = self.out(h)

        return out

from torch_geometric.transforms import AddLaplacianEigenvectorPE
class Performer(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [hidden_size] * (num_layers+1)
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        self.node_embedding = torch.nn.Linear(max_arity+3, hidden_size)
        self.input_norm = torch.nn.LayerNorm(hidden_size)
        #self.pe_embedding = torch.nn.Linear(5, hidden_size)
        #self.pe_norm = torch.nn.BatchNorm1d(hidden_size)
        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(GPSConv(self.hidden_sizes[i], None, heads=heads, attn_type='performer', dropout=dropout))
        self.out = torch.nn.Sequential(torch.nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]*2),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(dropout),
                                      torch.nn.Linear(self.hidden_sizes[-1]*2, 1))

        print("\n")
        print("LAYERS: ", len(self.layers))
        print("\n")

    def forward(self, data):
        #data = AddLaplacianEigenvectorPE(k=5, attr_name='pe')(data)
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        #pe = data.pe.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        x = self.node_embedding(x)
        x = self.input_norm(x)
        #pe = self.pe_embedding(pe)
        #pe = self.pe_norm(pe)
        #x = x + pe
        for i in range(len(self.layers)):
            x = self.layers[i](x=x, edge_index=edge_index, batch=batch)

        x = self.pool(x, batch)
        x = self.out(x)
        return x

class Transformer(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [hidden_size] * (num_layers+1)
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        self.node_embedding = torch.nn.Linear(max_arity+3, hidden_size)
        self.input_norm = torch.nn.LayerNorm(hidden_size)
        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(GPSConv(self.hidden_sizes[i], None, heads=heads, attn_type='multihead', dropout=dropout))
        self.out = torch.nn.Sequential(torch.nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]*2),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(dropout),
                                      torch.nn.Linear(self.hidden_sizes[-1]*2, 1))

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        x = self.node_embedding(x)
        x = self.input_norm(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x=x, edge_index=edge_index, batch=batch)

        x = self.pool(x, batch)
        x = self.out(x)
        return x

from torch_geometric.nn import GPSConv
class GCNGPS(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, dropout: int, heads: int, max_arity: int, pool, **kwargs):
        super().__init__()
        self.hidden_sizes = [hidden_size] * (num_layers+1)
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.pool = pool
        self.training = True

        self.node_embedding = torch.nn.Linear(max_arity+3, hidden_size)
        self.input_norm = torch.nn.LayerNorm(hidden_size)
        for i in range(len(self.hidden_sizes)-1):
            conv = GCNConv(self.hidden_sizes[i], self.hidden_sizes[i + 1])
            self.layers.append(GPSConv(self.hidden_sizes[i], conv, heads=heads, attn_type='performer', dropout=dropout))
        self.out = torch.nn.Linear(self.hidden_sizes[-1], 1)

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch
        if batch is not None:
            batch = batch.to(self.device)

        x = self.node_embedding(x)
        x = self.input_norm(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x=x, edge_index=edge_index, batch=batch)

        x = self.pool(x, batch)
        x = self.out(x)
        return x
