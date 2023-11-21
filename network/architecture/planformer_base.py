import torch
import torch.nn as nn
import pytorch_lightning as pl

# Imports related to type annotations
from typing import List, Dict, Tuple
from torch.nn.functional import Tensor


class RelationMessagePassing(nn.Module):
    def __init__(self, relations: List[Tuple[int, int]], hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.relation_modules = nn.ModuleList()
        for relation, arity in relations:  # TODO: REUSE THIS FOR READOUT
            assert relation == len(self.relation_modules)
            input_size = arity * hidden_size
            output_size = arity * hidden_size
            if (input_size > 0) and (output_size > 0):
                mlp = nn.Sequential(nn.Linear(input_size, input_size, True), nn.ReLU(), nn.Linear(input_size, output_size, True))
            else:
                mlp = None
            self.relation_modules.append(mlp)
        self.update = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size, True), nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size, True))
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    """
    def forward(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tuple[Tensor, Tensor]:
        # Compute an aggregated message for each recipient
        sum_msg = torch.zeros_like(node_states, dtype=torch.float, device=self.get_device())
        for relation, module in enumerate(self.relation_modules):
            if (module is not None) and (relation in relations):
                values = relations[relation].to(self.get_device())
                input = torch.index_select(node_states, 0, values).view(-1, module[0].in_features)
                output = module(input).view(-1, self.hidden_size)
                node_indices = values.view(-1, 1).repeat(1, self.hidden_size)
                sum_msg = torch.scatter_add(sum_msg, 0, node_indices, output)

        # Update states with aggregated messages
        next_node_states = self.update(torch.cat([sum_msg, node_states], dim=1))
        return next_node_states
    """
    def forward(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tensor:
        # Compute an aggregated message for each recipient
        max_outputs = []
        outputs = []
        for relation, module in enumerate(self.relation_modules):
            if (module is not None) and (relation in relations):
                values = relations[relation].to(self.get_device())
                input = torch.index_select(node_states, 0, values).view(-1, module[0].in_features)
                output = module(input).view(-1, self.hidden_size)
                max_outputs.append(torch.max(output))
                node_indices = values.view(-1, 1).expand(-1, self.hidden_size)
                outputs.append((output, node_indices))

        max_offset = torch.max(torch.stack(max_outputs))
        exps_sum = torch.full_like(node_states, 1E-16, device=self.get_device())
        for output, node_indices in outputs:
            exps = torch.exp(8.0 * (output - max_offset))
            exps_sum = torch.scatter_add(exps_sum, 0, node_indices, exps)

        # Update states with aggregated messages
        max_msg = ((1.0 / 8.0) * torch.log(exps_sum)) + max_offset
        next_node_states = self.update(torch.cat([max_msg, node_states], dim=1))
        return next_node_states


# TODO: REPLACE THIS WITH TRANSFORMER
class Readout(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, drop: float, n_layers: int, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.post_solvable = nn.Linear(d_model, self.output_size, bias)
        self.post_value = nn.Linear(d_model, self.output_size, bias)
        self.gnn2transformer = nn.Linear(self.input_size, d_model, bias)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=d_ff, dropout=drop)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.norm_input = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=n_layers, norm=self.encoder_norm)
        self.dummy = nn.Parameter(torch.empty(0))
        #  TODO: SKIP CONNECTION AROUND TRANSFORMER????
        # TODO: TRY REPLACING TRANSFORMER BY JUST AN MLP AND LOOK AT RESULTS

    def get_device(self):
        return self.dummy.device

    def forward(self, batch_num_objects: List[int], node_states: Tensor) -> Tensor:
        #print("\n")
        #print("node_states")
        #print(node_states.shape)

        # project node states to transformer dimensionality
        node_states = self.gnn2transformer(node_states)
        # normalize
        node_states = self.norm_input(node_states)

        #print("\n")
        #print("node_states")
        #print(node_states.shape)

        # split node states into batches
        node_states_batched = list(node_states.split(batch_num_objects))
        # find largest batch in node_states
        max_batch_size = max(batch_num_objects)
        # pad batches with zeros
        for i in range(len(node_states_batched)):
            node_states_batched[i] = torch.cat((node_states_batched[i], torch.zeros((max_batch_size - batch_num_objects[i], node_states.shape[1]), dtype=torch.float, device=self.get_device())), dim=0)
        # convert into single tensor
        node_states_batched = torch.stack(node_states_batched, dim=0).to(self.get_device())

        #print("\n")
        #print("node_states_batched")
        #print(node_states_batched.shape)


        # add classification token to each batch
        classification_token = torch.zeros((len(batch_num_objects), 1, node_states.shape[1]), dtype=torch.float, device=self.get_device())
        node_states_batched = torch.cat((classification_token, node_states_batched), dim=1)

        #print("\n")
        #print("node_states_batched")
        #print(node_states_batched.shape)

        # compute src_key_padding_mask
        src_key_padding_mask = torch.zeros((len(batch_num_objects), max_batch_size + 1), dtype=torch.bool, device=self.get_device())
        for i in range(len(batch_num_objects)):
            src_key_padding_mask[i, batch_num_objects[i]+1:] = True  # TODO: IS THIS CORRECT?
            #print("\n")
            #print(torch.sum(src_key_padding_mask[i]))
            #print(max_batch_size+1)
            #print(batch_num_objects[i]+1)
            assert torch.sum(src_key_padding_mask[i]) == (max_batch_size+1) - (batch_num_objects[i]+1)


        #print("\n")
        #print("src_key_padding_mask")
        #print(src_key_padding_mask.shape)


        # feed to transformer encoder
        transformer_encoding = self.transformer_encoder(node_states_batched, src_key_padding_mask=src_key_padding_mask)
        #print("\n")
        #print("transformer_encoding")
        #print(transformer_encoding.shape)

        classification_encoding = transformer_encoding[:, 0, :]  # TODO: IS THIS CORRECT?
        #print("\n")
        #print("classification_encoding")
        #print(classification_encoding.shape)

        post_value = self.post_value(classification_encoding)
        post_solvable = self.post_solvable(classification_encoding)

        #print("\n")
        #print("post_value")
        #print(post_value.shape)
        #print(post_value)
        #print("post_solvable")
        #print(post_solvable.shape)
        #print(post_solvable)

        #assert True == False
        return post_value, post_solvable

    def feature_vectors(self, batch_num_objects: List[int], node_states: Tensor) -> Tensor:
        results: List[Tensor] = []
        offset: int = 0
        nodes: Tensor = self.pre(node_states)
        for num_objects in batch_num_objects:
            intermediate = []
            intermediate.append(torch.sum(nodes[offset:(offset + num_objects)], dim=0))
            for layer in self.post:
                intermediate.append(layer(intermediate[-1]))
            results.append(torch.cat(intermediate))
            offset += num_objects
        return torch.stack(results)


class RelationMessagePassingModel(nn.Module):
    def __init__(self, relations: list, hidden_size: int, iterations: int, d_model: int, n_heads: int, d_ff: int,
                 drop: float, n_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.iterations = iterations
        self.relation_network = RelationMessagePassing(relations, hidden_size)
        self.readout = Readout(d_model, n_heads, d_ff, drop, n_layers, hidden_size, 1, True)
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]):
        node_states = self._initialize_nodes(sum(states[1]))
        node_states = self._pass_messages(node_states, states[0])
        value, solvable = self.readout(states[1], node_states)
        return value, solvable

    def feature_vectors(self, states: Tuple[Dict[int, Tensor], List[int]]):
        node_states = self._initialize_nodes(sum(states[1]))
        node_states = self._pass_messages(node_states, states[0])
        value = self.value_readout.feature_vectors(states[1], node_states)
        solvable = self.value_readout.feature_vectors(states[1], node_states)   # TODO: these two are identical?
        return value, solvable

    def _pass_messages(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tuple[Tensor, Tensor]:
        for _ in range(self.iterations):
             node_states = self.relation_network(node_states, relations)
        return node_states

    def _initialize_nodes(self, num_objects: int) -> Tensor:
        #print("\n")
        #print("num_objects")
        #print(num_objects)
        init_zeroes = torch.zeros((num_objects, (self.hidden_size // 2) + (self.hidden_size % 2)), dtype=torch.float, device=self.get_device())
        init_random = torch.randn((num_objects, self.hidden_size // 2), device=self.get_device())
        init_nodes = torch.cat([init_zeroes, init_random], dim=1)
        return init_nodes

#  TODO: STORE PREDICATES
class PlanFormerModelBase(pl.LightningModule):
    def __init__(self, predicates: List[Tuple[str, int]], hidden_size: int, iterations: int, d_model: int, n_heads: int, d_ff: int,
                 drop: float, n_layers: int):
        super().__init__()
        self.save_hyperparameters()
        encoding = dict([(predicate, index) for index, (predicate, _) in enumerate(predicates)])
        arities = [(encoding[predicate], arity) for predicate, arity in predicates]
        self.encoding = encoding
        self.model = RelationMessagePassingModel(arities, hidden_size, iterations, d_model, n_heads, d_ff, drop, n_layers)

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]):
        encoded_states = (dict([(self.encoding[name], values) for name, values in states[0].items()]), states[1])
        #print("\n")
        #print("Encoding")
        #print(self.encoding)
        #print("Encoded States")
        #print(encoded_states)
        #print(type(encoded_states))

        value, solvable = self.model(encoded_states)
        return torch.abs(value), solvable  # TODO: WHY IS HERE ABS(VALUE)?
