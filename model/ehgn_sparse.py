import torch

from Ex2.model.base import *
from torch_sparse import spmm
from torch_sparse import SparseTensor

# Sparse hierarchical network for the graphs with different numbers of nodes
class EHGN_TopK_Sparse(nn.Module):
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
        super(EHGN_TopK_Sparse, self).__init__()
        self.egnn_in = EGNN(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.egnn_hidden = EGNN(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.egnn_out = EGNN(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.poolingnn = TopKPooling(node_dim, ratio=cluster)
        self.mlp = MLP([node_dim, hidden_dim, hidden_dim, vector_dim])
        self.cluster = cluster

    def forward(self, edge_index, edge_attr, x, h, batch_idx):
        '''
        :param edge_index: adjacency matrix of shape (2, num_edges)
        :param edge_attr: edge features of shape (num_edges, edge_dim)
        :param x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
        :param h: node features of shape (num_nodes, node_dim)
        :param batch_idx:
        :return: result_x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
                 result_h: node features of shape (num_nodes, node_dim)
        '''
        x_l, h_l = x, h
        x, h = self.egnn_in(edge_index, edge_attr, x, h) # [bn, vector_dim], [bn, node_dim]

        # Derive high level information of x, h
        H, edge_index_a, edge_attr_a, batch_idx_a, perm, score = self.poolingnn(
            h, edge_index, edge_attr, batch=batch_idx
        ) # H: [bcluster, node_dim]
        X = x[perm] * score.view([-1, 1]) # X: [bcluster, vector_dim]
        X, H = self.egnn_hidden(edge_index_a, edge_attr_a, X, H) # [bcluster, vector_dim], [bcluster, node_dim]

        # Update low level information of X, H
        Xx, Hh = knn_interpolate(x=H, y=h_l, pos_x=X, pos_y=x_l,
                                 batch_x=batch_idx_a, batch_y=batch_idx,
                                 k=self.cluster, num_workers=1) # [bn, vector_dim], [bn, node_dim]
        result_x, result_h = self.egnn_out(edge_index, edge_attr, (x_l+Xx)/2, (h_l+Hh/2)) # [bn, vector_dim], [bn, node_dim]

        result_x = ((x_l - Xx) / 2) * self.mlp(result_h) + Xx
        return result_x, result_h

class DynamicsEHGN_TopK_Sparse(nn.Module):
    def __init__(self,
                 node_dim, edge_dim,
                 vector_dim, device,
                 model=EHGN_TopK_Sparse,
                 num_layers_egnn=4,
                 message_dim=128, hidden_dim=64,
                 scalar_dim=16, cluster=5):
        super(DynamicsEHGN_TopK_Sparse, self).__init__()
        self.time_embedding = FourierTimeEmbedding(embed_dim=hidden_dim, input_dim=1, device=device)
        self.mlp = MLP([hidden_dim, node_dim])
        self.model = model(num_layers_egnn, node_dim, edge_dim, message_dim, hidden_dim,
                           vector_dim, scalar_dim, cluster=cluster)
        # Sanity check
        self.invariant_scalr = MLP([node_dim, 1])

    def forward(self, t, edge_index, edge_attr, x, h, batch_idx):
        t = self.mlp(self.time_embedding(t))
        h = h + t
        x, h = self.model(edge_index, edge_attr, x, h, batch_idx)
        h = self.invariant_scalr(h)
        return x, h

    def reset_parameters(self):
        # Custom logic to reset or initialize parameters
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

# See <https://arxiv.org/abs/1706.02413>`_ paper.
def knn_interpolate(x, y, pos_x, pos_y,
                    batch_x=None, batch_y=None,
                    k=5, num_workers=1):
    '''
    :param x:
    :param y:
    :param pos_x:
    :param pos_y:
    :param batch_x:
    :param batch_y:
    :param k:
    :param num_workers:
    :return:
    '''
    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k,
                           batch_x=batch_x, batch_y=batch_y,
                           num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        diff_attr = x[x_idx] - y[y_idx]
        DIFF = torch.cat([diff, diff_attr], dim=-1)
        squared_distance = (DIFF * DIFF).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y_pos = scatter(pos_x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce='sum')
    y_pos = y_pos / scatter(weights, y_idx, 0, pos_y.size(0), reduce='sum')

    y_attr = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce='sum')
    y_attr = y_attr / scatter(weights, y_idx, 0, pos_y.size(0), reduce='sum')
    return y_pos, y_attr

if __name__ == '__main__':
    config = parse_toml_file('/home/she0000/PycharmProjects/pythonProject/Ex2/config.toml')
    directory_path = config['directory_path']
    data_dir = config['data_dir']
    dataset_location = os.path.join(data_dir, 'dataset.pickle')
    cutoff = config['cutoff']
    scale = config['scale']
    node_dim = config['node_dim']
    edge_dim = config['edge_dim']
    vector_dim = config['vector_dim']
    device = config['device']

    # Create your dataset instance
    TrajsDataset = TrajectoriesDataset(
        directory_path,
        cutoff=cutoff,
        scale=scale
    )

    pickle_object(TrajsDataset, dataset_location)
    dataset = unpickle_object(dataset_location)
    batch_size, train_loader, val_loader = generate_loaders(dataset, config)

    dynamicsehgn_topk_sparse = DynamicsEHGN_TopK_Sparse(node_dim, edge_dim, vector_dim, device).to(device)
    for batch in train_loader:
        batch = augment_batch(batch).to(device)
        batch_idx = batch.batch
        out_x, out_h = dynamicsehgn_topk_sparse(t=batch.frame_idx,
                                                edge_index=batch.edge_index,
                                                edge_attr=batch.edge_encoding,
                                                x=batch.pos,
                                                h=batch.x,
                                                batch_idx=batch_idx)
        print(out_x.shape, out_h.shape)
        break
