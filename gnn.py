import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import HierarchicalNode, GlobalNode

from torch_scatter import scatter_mean


class GlobalGNN(torch.nn.Module):

    def __init__(self, num_tasks=1, gnn='gin', num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last",
                 graph_pooling="sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GlobalGNN, self).__init__()

        self.gnn = gnn
        self.num_layers = num_layers
        self.residual = residual
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.JK == 'concat':
            self.pred_in_channels = self.num_layers * self.emb_dim
        elif self.JK == 'last' or self.JK == 'sum':
            self.pred_in_channels = self.emb_dim
        else:
            raise ValueError("Invalid Jump Knowledge concatenation type.")

        # GNN to generate node embeddings
        self.gnn_node = GlobalNode(gnn=self.gnn, num_layers=self.num_layers, emb_dim=self.emb_dim,
                                   drop_ratio=self.drop_ratio, residual=self.residual)

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.pred_in_channels, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.pred_in_channels, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        # Different implementations of Jk-concat
        if self.JK == 'concat':
            h_graph = torch.tensor([], device=h_node[0].device)
            for layer in range(self.num_layers):
                h = self.pool(h_node[layer + 1], batched_data.batch)
                h_graph = torch.cat((h_graph, h), dim=1)
        elif self.JK == 'sum':
            node_representation = torch.tensor([], device=h_node[0].device)
            for layer in range(self.num_layers):
                node_representation += h_node[layer + 1]
            h_graph = self.pool(node_representation, batched_data.batch)
        else:  # self.JK = 'last'
            h_graph = self.pool(h_node[-1], batched_data.batch)

        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)  # 限定HOMO-LUMO gap在(0,50)之间


class HierarchicalGNN(torch.nn.Module):
    """
    define a GNN use virtual node embedding as graph embedding.
    Pooling funciton is defined in MyNode.
    """

    def __init__(self, num_tasks=1, gnn='gin', num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last",
                 graph_pooling="sag"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(HierarchicalGNN, self).__init__()

        self.gnn = gnn
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.residual = residual
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.JK == 'concat':
            self.pred_in_channels = self.num_layers * self.emb_dim
        elif self.JK == 'last' or self.JK == 'sum':
            self.pred_in_channels = self.emb_dim
        else:
            raise ValueError("Invalid Jump Knowledge concatenation type.")

        # GNN to generate node embeddings
        self.gnn_node = HierarchicalNode(num_layers=self.num_layers, gnn=self.gnn,
                                         emb_dim=self.emb_dim, drop_ratio=drop_ratio,
                                         graph_pooling=self.graph_pooling,
                                         residual=self.residual)
        # Only use Sum_Pooling for hierarchical node embeddings.
        self.pool = global_add_pool

        self.graph_pred_linear = torch.nn.Linear(self.pred_in_channels, self.num_tasks)

    def forward(self, batched_data):
        # get virtual node embedding batch that after pooling
        h_node, b_list = self.gnn_node(batched_data)

        # Different implementations of Jk-concat
        if self.JK == 'concat':
            h_graph = torch.tensor([], device=h_node[0].device)
            for layer in range(self.num_layers):
                h = self.pool(h_node[layer + 1], b_list[layer + 1])
                h_graph = torch.cat((h_graph, h), dim=1)
        elif self.JK == 'sum':
            h_graph = torch.tensor([], device=h_node[0].device)
            for layer in range(self.num_layers):
                h_graph += self.pool(h_node[layer + 1], b_list[layer + 1])
        else:  # self.JK = 'last'
            h_graph = self.pool(h_node[-1], b_list[-1])

        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)  # 限定HOMO-LUMO gap在(0,50)之间


if __name__ == '__main__':
    GlobalGNN(num_tasks=10)
