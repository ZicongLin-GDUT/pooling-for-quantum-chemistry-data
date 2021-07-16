import torch
from torch import Tensor
from torch_geometric.nn.pool.topk_pool import filter_adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.glob import global_add_pool
from torch_geometric.nn.conv import GCNConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn.pool import SAGPooling, TopKPooling, ASAPooling, PANPooling
from torch_geometric.utils import degree


# GCN convolution along the graph structure
class GCN(MessagePassing):
    def __init__(self, emb_dim):
        super(GCN, self).__init__(aggr="add")
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        row, col = edge_index
        # Because edge_attr is used in message functions, cannot add edges directly using 'add_self_loop'
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


# GIN convolution along the graph structure
class GIN(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GIN, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))  # Epsilon

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))  # GIN formula

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# Generate node embedding using global pooling.
class GlobalNode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, gnn='gin', drop_ratio=0.5, residual=False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GlobalNode, self).__init__()
        self.gnn = gnn
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        # set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        # batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        # List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            if self.gnn == 'gin':
                self.convs.append(GIN(emb_dim))
            elif self.gnn == 'gcn':
                self.convs.append(GCN(emb_dim))
            else:
                raise ValueError('Invalid GNN type')
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        edge_attr = self.bond_encoder(edge_attr)

        # virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            # add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # update the virtual nodes
            if layer < self.num_layers - 1:
                # add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding

                # transform virtual nodes using MLP
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                      self.drop_ratio, training=self.training)

        return h_list


# Generate node embedding using hierarchical pooling.
class HierarchicalNode(torch.nn.Module):
    """
    Use GCNConv to compute score.
    Cannot use residual for node embedding because of the change of graph, only use for virtual node embedding.
    """

    def __init__(self, num_layers, emb_dim, gnn='gin', drop_ratio=0.5, residual=False, graph_pooling='sag'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(HierarchicalNode, self).__init__()
        self.gnn = gnn
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.residual = residual
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        # set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # hierarchical pooling methods
        if self.graph_pooling == 'sag':
            self.hierarchical_pooling = SAGPooling(in_channels=emb_dim, GNN=GCNConv, ratio=0.5)
        elif self.graph_pooling == 'gpool':
            self.hierarchical_pooling = TopKPooling(in_channels=emb_dim, ratio=0.5)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        # batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()
        # List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        # List of SAGPooling to sample input graphs.
        self.pool = torch.nn.ModuleList()

        for layer in range(num_layers):
            if self.gnn == 'gin':
                self.convs.append(GIN(emb_dim))
            elif self.gnn == 'gcn':
                self.convs.append(GCN(emb_dim))
            else:
                raise ValueError('Invalid GNN type')
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
            self.pool.append(self.hierarchical_pooling)

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        edge_attr = self.bond_encoder(edge_attr)

        # virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        b_list = [batch]
        for layer in range(self.num_layers):
            # add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            # update the virtual nodes
            if layer < self.num_layers - 1:
                # add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                # transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                      self.drop_ratio, training=self.training)
            # hierarchical pooling
            if layer < self.num_layers - 1:
                if self.graph_pooling == 'sag' or self.graph_pooling == 'gpool':
                    h, edge_index, edge_attr, batch, _, _ = self.pool[layer](x=h, edge_index=edge_index,
                                                                             edge_attr=edge_attr,
                                                                             batch=batch, attn=None)
                elif self.graph_pooling == 'asap':
                    h, _, _, batch, perm = self.pool[layer](x=h, edge_index=edge_index, edge_weight=None, batch=batch)
                    edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm)

            h_list.append(h)
            b_list.append(batch)
        return h_list, b_list
