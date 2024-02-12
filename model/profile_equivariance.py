from Ex2.model.ehgn_sparse import *
import torch
from torch import sin, cos, atan2, acos

def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

def rot_y(beta):
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def measure_gnn_sparse_equivariance(model, inputs):
    time = inputs[0]
    edge_index = inputs[1]
    edge_encoding = inputs[2]
    pos = inputs[3]
    x = inputs[4]

    R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float).to(pos.device)
    T = torch.randn(1, 3).to(pos.device)
    apply_action = lambda a: (a.to(torch.float) @ R + T)

    # Cache first two nodes' features
    node1 = x[0, :]
    node2 = x[1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = x.clone().detach()
    feats_permuted_row_wise[0, :] = node2
    feats_permuted_row_wise[1, :] = node1

    pos_new = apply_action(pos)

    x1, h1 = model(time, edge_index, edge_encoding, pos, x)
    x2, h2 = model(time, edge_index, edge_encoding, pos_new, x)
    x3, h3 = model(time, edge_index, edge_encoding, pos_new, feats_permuted_row_wise)

    print(h1 - h2)
    print(apply_action(x1) - x2)
    print(h1 - h3)
    print(torch.allclose(h1, h2))
    print(torch.allclose(apply_action(x1), x2))
    print(torch.allclose(h1, h3, atol=1e-6))
    assert torch.allclose(h1, h2), 'features must be invariant'
    assert torch.allclose(apply_action(x1), x2), 'coordinates must be equivariant'
    assert not torch.allclose(h1, h3, atol=1e-6), 'model must be equivariant to permutations of node order'

def measure_ehgn_sparse_equivariance(model, inputs):
    time = inputs[0]
    edge_index = inputs[1]
    edge_encoding = inputs[2]
    pos = inputs[3]
    x = inputs[4]
    batch_idx = inputs[5]

    R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float).to(pos.device)
    T = torch.rand(1, 3).to(pos.device)
    apply_action = lambda a: (a.to(torch.float) @ R + T)

    # Cache first two nodes' features
    node1 = x[0, :]
    node2 = x[1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = x.clone().detach()
    feats_permuted_row_wise[0, :] = node2
    feats_permuted_row_wise[1, :] = node1

    pos_new = apply_action(pos)

    x1, h1 = model(time, edge_index, edge_encoding, pos, x, batch_idx)
    x2, h2 = model(time, edge_index, edge_encoding, pos_new, x, batch_idx)
    x3, h3 = model(time, edge_index, edge_encoding, pos_new, feats_permuted_row_wise, batch_idx)

    print(h1 - h2)
    print(apply_action(x1) - x2)
    print(h1 - h3)
    print(torch.allclose(h1, h2))
    print(torch.allclose(apply_action(x1), x2))
    print(torch.allclose(h1, h3, atol=1e-6))
    assert torch.allclose(h1, h2), 'features must be invariant'
    assert torch.allclose(apply_action(x1), x2), 'coordinates must be equivariant'
    assert not torch.allclose(h1, h3, atol=1e-6), 'model must be equivariant to permutations of node order'

def measure_gnn_sparse_equivariance_simple(model, device):
    time = torch.full((16, 1), 1).to(device)
    edge_index = (torch.rand(2, 20) * 16).long().to(device)
    edge_encoding = torch.randn(20, 4).to(device)
    pos = torch.randn(16, 3).to(device)
    x = torch.randn(16, 1).to(device)

    R = rot(*torch.rand(3)).to(device)
    T = torch.randn(1, 3).to(device)
    apply_action = lambda a: (a @ R + T)

    # Cache first two nodes' features
    node1 = x[0, :]
    node2 = x[1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = x.clone().detach()
    feats_permuted_row_wise[0, :] = node2
    feats_permuted_row_wise[1, :] = node1

    pos_new = apply_action(pos)

    x1, h1 = model(time, edge_index, edge_encoding, pos, x)
    x2, h2 = model(time, edge_index, edge_encoding, pos_new, x)
    x3, h3 = model(time, edge_index, edge_encoding, pos_new, feats_permuted_row_wise)

    print(h1 - h2)
    print(apply_action(x1) - x2)
    print(h1 - h3)
    print(torch.allclose(h1, h2))
    print(torch.allclose(apply_action(x1), x2))
    print(torch.allclose(h1, h3, atol=1e-6))
    assert torch.allclose(h1, h2), 'features must be invariant'
    assert torch.allclose(apply_action(x1), x2), 'coordinates must be equivariant'
    assert not torch.allclose(h1, h3, atol=1e-6), 'model must be equivariant to permutations of node order'

def measure_ehgn_sparse_equivariance_simple(model, device):
    time = torch.full((16, 1), 1).to(device)
    edge_index = (torch.rand(2, 20) * 16).long().to(device)
    edge_encoding = torch.randn(20, 4).to(device)
    pos = torch.randn(16, 3).to(device)
    x = torch.randn(16, 1).to(device)
    batch_idx = torch.full((16, ), 0).to(device)

    R = rot(*torch.rand(3)).to(device)
    T = torch.randn(1, 3).to(device)
    apply_action = lambda a: (a @ R + T)

    # Cache first two nodes' features
    node1 = x[0, :]
    node2 = x[1, :]

    # Switch first and second nodes' positions
    feats_permuted_row_wise = x.clone().detach()
    feats_permuted_row_wise[0, :] = node2
    feats_permuted_row_wise[1, :] = node1

    pos_new = apply_action(pos)

    x1, h1 = model(time, edge_index, edge_encoding, pos, x, batch_idx)
    x2, h2 = model(time, edge_index, edge_encoding, pos_new, x, batch_idx)
    x3, h3 = model(time, edge_index, edge_encoding, pos_new, feats_permuted_row_wise, batch_idx)

    print(h1 - h2)
    print(apply_action(x1) - x2)
    print(h1 - h3)
    print(torch.allclose(h1, h2))
    print(torch.allclose(apply_action(x1), x2))
    print(torch.allclose(h1, h3, atol=1e-6))
    assert torch.allclose(h1, h2), 'features must be invariant'
    assert torch.allclose(apply_action(x1), x2), 'coordinates must be equivariant'
    assert not torch.allclose(h1, h3, atol=1e-6), 'model must be equivariant to permutations of node order'


if __name__ == "__main__":
    config = parse_toml_file('/home/she0000/PycharmProjects/pythonProject/Ex2/config.toml')
    directory_path = config['directory_path']
    data_dir = config['data_dir']
    dataset_location = os.path.join(data_dir, 'dataset.pickle')
    cutoff = config['cutoff']
    scale = config['scale']
    node_dim = config['node_dim']
    edge_dim = config['edge_dim']
    vector_dim = config['vector_dim']
    device0 = config['device0']
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

    model1 = DynamicsGNN(node_dim, edge_dim, vector_dim, device0).to(device0)
    model2 = DynamicsEGNN(node_dim, edge_dim, vector_dim, device0).to(device0)
    model3 = DynamicsEHGN_TopK_Sparse(node_dim, edge_dim, vector_dim, device0).to(device0)

    measure_gnn_sparse_equivariance_simple(model1, device=device0)
    print('***************************************************************************')
    measure_gnn_sparse_equivariance_simple(model2, device=device0)
    print('***************************************************************************')
    measure_ehgn_sparse_equivariance_simple(model3, device=device0)
    print('***************************************************************************')

    # for batch in train_loader:
    #     batch = augment_batch(batch).to(device0)
    #     inputs = (batch.frame_idx, batch.edge_index, batch.edge_encoding, batch.pos, batch.x, batch.batch)
    #     measure_gnn_sparse_equivariance(model1, inputs)
    #     print('***************************************************************************')
    #     measure_gnn_sparse_equivariance(model2, inputs)
    #     print('***************************************************************************')
    #     measure_ehgn_sparse_equivariance(model3, inputs)
    #     print('***************************************************************************')
    #     break
    #
