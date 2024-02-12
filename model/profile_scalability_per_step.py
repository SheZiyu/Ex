# Metrics to compare time complexity (speed of inference) and space complexity (model size)

# Time complexity in terms of FLOPs (floating-point operations)
# FLOPs are used to describe how many operations are required to run a single instance of a given model,
# the more the FLOPs the more time model will take for inference

# Model size in terms of the number of parameters

# Latency is the amount of time it takes for a neural network to produce a prediction for a single input sample

# Throughput is the number of predictions produced by a neural network in a given amount of time
from Ex2.model.ehgn_sparse import *
import time
import psutil
import torch

# Latency is the amount of time it takes for a neural network to produce a prediction for a single input sample
def measure_latency_cpu_usage(model, test_inputs):
    process = psutil.Process()
    cpu_start = process.cpu_percent()
    start = time.time()
    x, h = model(*test_inputs)
    end = time.time()
    cpu_end = process.cpu_percent()
    latency = end - start
    cpu_usage = cpu_end - cpu_start
    return latency, cpu_usage

# Throughput is the number of predictions produced by a neural network in a given amount of time
def measure_gpu_throughput(model, batch_size, test_inputs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        x, h = model(*test_inputs)
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end)
    throughput = batch_size / latency
    return throughput

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

    model1 = DynamicsGNN(node_dim, edge_dim, vector_dim, device).to(device)
    model2 = DynamicsEGNN(node_dim, edge_dim, vector_dim, device).to(device)
    model3 = DynamicsEHGN_TopK_Sparse(node_dim, edge_dim, vector_dim, device).to(device)
    for batch in train_loader:
        batch = augment_batch(batch).to(device)
        batch_idx = batch.batch
        latency, cpu_usage = measure_latency_cpu_usage(
            model1, (batch.frame_idx,
                     batch.edge_index,
                     batch.edge_encoding,
                     batch.pos,
                     batch.x)
        )
        print('GNN: Latency {}, CPU_usage {} per batch'.format(latency, cpu_usage))
        print('***************************************************************************')
        latency, cpu_usage = measure_latency_cpu_usage(
            model2, (batch.frame_idx,
                     batch.edge_index,
                     batch.edge_encoding,
                     batch.pos,
                     batch.x)
        )
        print('EGNN: Latency {}, CPU_usage {} per batch'.format(latency, cpu_usage))
        print('***************************************************************************')
        latency, cpu_usage = measure_latency_cpu_usage(
            model3, (batch.frame_idx,
                     batch.edge_index,
                     batch.edge_encoding,
                     batch.pos,
                     batch.x,
                     batch_idx)
        )
        print('EHGN: Latency {}, CPU_usage {} per batch'.format(latency, cpu_usage))
        print('***************************************************************************')
        break

    model1 = DynamicsGNN(node_dim, edge_dim, vector_dim, device0).to(device0)
    model2 = DynamicsEGNN(node_dim, edge_dim, vector_dim, device0).to(device0)
    model3 = DynamicsEHGN_TopK_Sparse(node_dim, edge_dim, vector_dim, device0).to(device0)
    for batch in train_loader:
        batch = augment_batch(batch).to(device0)
        batch_idx = batch.batch
        noised_pos = batch.pos + torch.randn_like(batch.pos)
        throughput = measure_gpu_throughput(model1, batch_size,
                                            (batch.frame_idx,
                                             batch.edge_index,
                                             batch.edge_encoding,
                                             noised_pos,
                                             batch.x))
        print('GNN: GPU_throughput {} per batch'.format(throughput))
        print('***************************************************************************')
        throughput = measure_gpu_throughput(model2, batch_size,
                                            (batch.frame_idx,
                                             batch.edge_index,
                                             batch.edge_encoding,
                                             noised_pos,
                                             batch.x))
        print('EGNN: GPU_throughput {} per batch'.format(throughput))
        print('***************************************************************************')
        throughput = measure_gpu_throughput(model3, batch_size,
                                            (batch.frame_idx,
                                             batch.edge_index,
                                             batch.edge_encoding,
                                             noised_pos,
                                             batch.x,
                                             batch_idx))
        print('EHGN: GPU_throughput {} per batch'.format(throughput))
        print(summary(model1, *(batch.frame_idx,
                                batch.edge_index,
                                batch.edge_encoding,
                                batch.pos,
                                batch.x)))
        print('***************************************************************************')
        print(summary(model2, *(batch.frame_idx,
                                batch.edge_index,
                                batch.edge_encoding,
                                batch.pos,
                                batch.x)))
        print('***************************************************************************')
        print(summary(model3, *(batch.frame_idx,
                                batch.edge_index,
                                batch.edge_encoding,
                                batch.pos,
                                batch.x,
                                batch_idx)))
        print('***************************************************************************')
        break

