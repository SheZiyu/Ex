import torch

from Ex2.model.base import *
from torch_sparse import spmm
from torch_sparse import SparseTensor

# Dense hierarchical network for the graphs with the same number of nodes
class EHGN_TopK_Dense(nn.Module):
    def __init__(self,
                 num_layers_egnn,
                 node_dim, edge_dim,
                 message_dim, hidden_dim,
                 vector_dim, scalar_dim,
                 cluster=5
                 ):
        '''
        :param num_layers_egnn:
        :param node_dim:
        :param edge_dim:
        :param message_dim:
        :param hidden_dim:
        :param vector_dim:
        :param scalar_dim:
        :param cluster: K number of clusters
        '''
        super(EHGN_TopK_Dense, self).__init__()
        self.egnn_in = EGNN(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.egnn_hidden = EGNN(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.egnn_out = EGNN(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.poolingnn = MLP([node_dim, hidden_dim*8, hidden_dim, cluster])
        self.mlp_1 = MLP([1, hidden_dim, hidden_dim, edge_dim])
        self.mlp_2 = MLP([node_dim, hidden_dim, hidden_dim, vector_dim])
        self.cluster = cluster

    def forward(self, edge_index, edge_attr, x, h, num_nodes, batch_size, node_mask=None):
        '''
        :param edge_index: adjacency matrix of shape (2, num_edges)
        :param edge_attr: edge features of shape (num_edges, edge_dim)
        :param x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
        :param h: node features of shape (num_nodes, node_dim)
        :param num_nodes:
        :param batch_size:
        :param node_mask: node mask of shape (num_nodes, ) when num_nodes are different in the batch
        :return: result_x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
                 result_h: node features of shape (num_nodes, node_dim)
        '''
        x_l, h_l = x, h
        x, h = self.egnn_in(edge_index, edge_attr, x, h)
        # Derive high level information of x, h
        pooling_attr = self.poolingnn(h) # [bn, cluster]
        hard_pooling = pooling_attr.argmax(dim=-1)
        hard_pooling = F.one_hot(hard_pooling, num_classes=self.cluster).float()
        self.current_pooling_plan = hard_pooling # [bn, cluster]

        pooling = F.softmax(pooling_attr, dim=-1) # [bn, cluster]
        s = pooling.reshape([-1, num_nodes//batch_size, pooling.shape[-1]]) # [b, n, cluster]
        s_t = s.transpose(-2, -1) # [b, cluster, n]
        p_index = torch.ones_like(x)[..., 0]
        if node_mask is not None:
            p_index = p_index * node_mask
        p_index = p_index.reshape([-1, num_nodes//batch_size, 1]) # [b, n, 1]
        count = torch.einsum('bij,bjk->bik', s_t, p_index).clamp_min(1e-5) # [b, cluster, 1]
        _x = x.reshape([-1, num_nodes//batch_size, x.shape[-1]]) # [b, n, vector_dim]
        _h_l = h_l.reshape([-1, num_nodes//batch_size, h_l.shape[-1]]) # [b, n, node_dim]
        X = torch.einsum('bij,bjk->bik', s_t, _x) # [b, cluster, vector_dim]
        H = torch.einsum('bij,bjk->bik', s_t, _h_l) # [b, cluster, node_dim]
        X = (X / count).reshape([-1, X.shape[-1]]) # [bcluster, vector_dim]
        H = (H / count).reshape([-1, H.shape[-1]]) # [bcluster, node_dim]

        # Derive high level adjacency matrix of shape (b, cluster, cluster)
        a = spmm(edge_index, torch.ones_like(edge_index[0]), x.shape[0], x.shape[0], pooling) # [bn, cluster]
        a = a.reshape([-1, num_nodes//batch_size, a.shape[-1]]) # [b, n, cluster]
        a = torch.einsum('bij,bjk->bik', s_t, a) # [b, cluster, cluster]
        row_a, col_a, edge_attr_a, edge_mask_a = construct_edges(a, self.cluster)
        edge_index_a = torch.stack([row_a, col_a], dim=0)
        edge_attr_a = self.mlp_1(edge_attr_a.reshape([-1, 1]))
        X, H = self.egnn_hidden(edge_index_a, edge_attr_a, X, H) # [bcluster, vector_dim], [bcluster, node_dim]

        # Update low level information of X, H
        _X = X.reshape([-1, self.cluster, X.shape[-1]])  # [b, cluster, vector_dim]
        _H = H.reshape([-1, self.cluster, H.shape[-1]])  # [b, cluster, node_dim]
        Xx = torch.einsum('bij,bjk->bik', s, _X)  # [b, n, vector_dim]
        Hh = torch.einsum('bij,bjk->bik', s, _H)  # [b, n, node_dim]
        Xx = Xx.reshape([-1, Xx.shape[-1]])  # [bn, vector_dim]
        Hh = Hh.reshape([-1, Hh.shape[-1]])  # [bn, node_dim]

        result_x, result_h = self.egnn_out(edge_index, edge_attr, (x_l+Xx)/2, (h_l+Hh/2)) # [bn, vector_dim], [bn, node_dim]
        result_x = ((x_l - Xx) / 2) * self.mlp_2(result_h) + Xx
        return result_x, result_h

class DynamicsEHGN_TopK_Dense(nn.Module):
    def __init__(self,
                 node_dim, edge_dim,
                 vector_dim, device,
                 model=EHGN_TopK_Dense,
                 num_layers_egnn=4,
                 message_dim=128, hidden_dim=64,
                scalar_dim=16, cluster=5):
        super(DynamicsEHGN_TopK_Dense, self).__init__()
        self.time_embedding = FourierTimeEmbedding(embed_dim=hidden_dim, input_dim=1, device=device)
        self.mlp = MLP([hidden_dim, node_dim])
        self.model = model(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim,
                           vector_dim, scalar_dim, cluster=cluster)
        # Sanity check
        self.invariant_scalr = MLP([node_dim, 1])

    def forward(self, t, edge_index, edge_attr, x, h, num_nodes, batch_size, node_mask=None):
        t = self.mlp(self.time_embedding(t))
        h = h + t
        x, h = self.model(edge_index, edge_attr, x, h, num_nodes=num_nodes, batch_size=batch_size, node_mask=node_mask)
        h = self.invariant_scalr(h)
        return x, h

    def reset_parameters(self):
        # Custom logic to reset or initialize parameters
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()



