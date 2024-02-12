'''
plot dx distribution to test the impact of scaling
'''
from Ex2.data.preprocessing import *

import numpy as np

from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import torch
from torch import nn
from torch import Tensor

from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph, summary, knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, to_networkx
import networkx as nx

import os
import zipfile
import MDAnalysis as mda
import numpy as np
import shutil

class ProteinAnalysis:
    def __init__(self, directory_path, selection='not (name H*) and name CA', scaling=True, scale=1.0):
        self.directory_path = directory_path
        self.selection = selection
        self.scaling = scaling
        self.scale = scale
        self.coordinates_arrays = []
        self.rmsf_values = []
        self.one_hot_trajs = []
        self.names = []

    def load_coordinate_rmsf_onehot(self):
        # Iterate over each protein system in the directory
        for zip_file_name in os.listdir(self.directory_path):
            if zip_file_name.endswith('.zip'):
                system_path = os.path.join(self.directory_path, zip_file_name)

                # Create a temporary directory with the same name as the zip file (without extension)
                temp_dir = os.path.splitext(zip_file_name)[0]
                temp_dir_path = os.path.join(self.directory_path, temp_dir)
                os.makedirs(temp_dir_path, exist_ok=True)

                # Extract contents of the zip file to the temporary directory
                with zipfile.ZipFile(system_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path)

                # Load true RMSF
                rmsf_path = self._find_rmsf_tsv_files(temp_dir_path)[0]
                rmsf_true = self._load_tsv_file(rmsf_path)

                # Append the true RMSF values to the list for the current replicate
                self.rmsf_values.extend(np.split(rmsf_true, rmsf_true.shape[1], axis=1))

                # Check if the directory contains trajectory and topology files
                xtc_files = [file for file in os.listdir(temp_dir_path) if file.endswith('.xtc')]
                if not xtc_files:
                    print('No XTC files found in {}'.format(temp_dir_path))
                pdb_file = [file for file in os.listdir(temp_dir_path) if file.endswith('.pdb')]
                if not pdb_file:
                    print('No PDB file found in {}'.format(temp_dir_path))

                # Load trajectory and topology using MDAnalysis
                for xtc_file in xtc_files:
                    trajectory_path = os.path.join(temp_dir_path, xtc_file)
                    topology_path = os.path.join(temp_dir_path, pdb_file[0])
                    self.names.append(xtc_file[0][:-4])
                    try:
                        u = mda.Universe(topology_path, trajectory_path)
                        one_hot_encoded_atoms = self._one_hot_encode_atoms(u, selection=self.selection)
                        self.one_hot_trajs.append(one_hot_encoded_atoms)
                    except Exception as e:
                        print('Error loading trajectory and topology: {}'.format(e))

                    # Initialize an empty array to store coordinates
                    coordinates_array = np.empty((len(u.trajectory), u.select_atoms(self.selection).n_atoms, 3))

                    # Iterate through the trajectory frames and store coordinates
                    for i, ts in enumerate(u.trajectory):
                        coordinates_array[i] = u.select_atoms(self.selection).positions
                    if self.scaling:
                        coordinates_array *= self.scale
                    self.coordinates_arrays.append(coordinates_array)

                    # Save the NumPy array to a file with a trajectory-specific name
                    output_filename = os.path.join(
                        self.directory_path,
                        'coordinates_{}_{}.npy'.format(self.selection.replace(' ', '_').lower(), xtc_file[0][:-4])
                    )
                    np.save(output_filename, coordinates_array)

                # Clean up: Remove the temporary extracted directory
                shutil.rmtree(temp_dir_path)

        return self.coordinates_arrays, self.rmsf_values, self.one_hot_trajs, self.names

    def _find_rmsf_tsv_files(self, directory):
        # Helper function to find RMSF TSV files in a directory
        search_pattern = '{}/*RMSF*.tsv'.format(directory)
        file_paths = glob.glob(search_pattern)
        return file_paths

    def _load_tsv_file(self, tsv_file_path):
        # Helper function to load data from a TSV file
        data_frame = pd.read_csv(tsv_file_path, sep='\t')
        data_array = data_frame.values
        return data_array[:, 1:]

    def _one_hot_encode_atoms(self, u, selection):
        # Helper function to one-hot encode atoms
        selected_atoms = u.select_atoms(selection)

        # Get the unique atom names
        unique_atom_names = np.unique(selected_atoms.names)

        # Create a dictionary mapping each atom name to its one-hot encoded index
        atom_to_index = {atom: i for i, atom in enumerate(unique_atom_names)}

        # Initialize an array to store the one-hot encoded representations
        one_hot_encoded = np.zeros((len(selected_atoms), len(unique_atom_names)))

        # Iterate through the selected atoms and set the corresponding one-hot encoded values
        for i, atom in enumerate(selected_atoms):
            atom_name = atom.name
            one_hot_encoded[i, atom_to_index[atom_name]] = 1
        return one_hot_encoded

class TrajectoriesDataset(Dataset):
    def __init__(
            self,
            directory_path,
            cutoff,
            scale=1.0,
            augment=True,
            dataset=[]
    ):
        super(TrajectoriesDataset, self).__init__()
        self.augment = augment
        self.dataset = dataset

        coordinates_arrays, rmsf_values, one_hot_trajs, names = \
            ProteinAnalysis(directory_path).load_coordinate_rmsf_onehot()

        for traj_idx, (coordinates_array, one_hot, name, rmsf_value) in enumerate(zip(coordinates_arrays, one_hot_trajs, names, rmsf_values)):
            one_hot = (torch.tensor(one_hot, dtype=torch.float)).unsqueeze(0).expand([coordinates_array.shape[0], -1, -1])
            rmsf_value = (torch.tensor(rmsf_value.astype(np.float_), dtype=torch.float)).unsqueeze(0).expand([coordinates_array.shape[0], -1, -1])

            # Instantiate dataset as a list of PyG Data objects
            for frame_idx, (coordinates_i, one_hot_i, rmsf_value_i) in tqdm(enumerate(zip(coordinates_array, one_hot, rmsf_value)), disable=True,
                                                                            desc='Sampling {}'.format(name), total=len(coordinates_array)):
                coordinates_i = torch.tensor(coordinates_i, dtype=torch.float)
                i, j = radius_graph(coordinates_i, r=cutoff * scale)
                frame_idx = torch.tensor(frame_idx).repeat([coordinates_i.shape[0], 1])
                data = Data(
                    pos=coordinates_i,
                    x=one_hot_i,
                    y=rmsf_value_i,
                    edge_index=torch.stack([i, j]),
                    traj_idx=traj_idx,
                    frame_idx=frame_idx
                )
                self.dataset.append(data)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        if self.augment:
            return self.random_rotate(self.dataset[idx])
        else:
            return self.dataset[idx]

    def random_rotate(self, data):
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float, device=data.pos.device)
        data.pos @= R
        return data

def construct_edges(A, n_node):
    # Flatten the adjacency matrix
    h_edge_fea = A.reshape(-1) # [BPP]

    # Create indices for row and column
    h_row = torch.arange(A.shape[1]).unsqueeze(-1).expand([-1, A.shape[1]]).reshape(-1).to(A.device)
    h_col = torch.arange(A.shape[1]).unsqueeze(0).expand([A.shape[1], -1]).reshape(-1).to(A.device)

    # Expand row and column indices for batch dimension
    h_row = h_row.unsqueeze(0).expand([A.shape[0], -1])
    h_col = h_col.unsqueeze(0).expand([A.shape[0], -1])

    # Calculate offset for batch-wise indexing
    offset = (torch.arange(A.shape[0]) * n_node).unsqueeze(-1).to(A.device)

    # Apply offset to row and column indices
    h_row, h_col = (h_row + offset).reshape(-1), (h_col + offset).reshape(-1)

    # Create an edge mask where diagonal elements are set to 0
    h_edge_mask = torch.ones_like(h_row)
    base_diag_indices = (torch.arange(A.shape[1]) * (A.shape[1] + 1)).to(A.device)
    diag_indices_tensor = torch.tensor([]).to(A.device)
    for i in range(A.shape[0]):
        diag_indices = base_diag_indices + i * A.shape[1] * A.shape[-1]
        diag_indices_tensor = torch.cat([diag_indices_tensor, diag_indices], dim=0).long()
    h_edge_mask[diag_indices_tensor] = 0

    return h_row, h_col, h_edge_fea, h_edge_mask

# PLot dx distribution
def plt_dx_distribution(num_frames_to_process, traj1, traj2):
    # Generate coordinates over time
    time = np.arange(0, num_frames_to_process - 1)

    # dx trajectory
    dx_trajectory1 = np.zeros([num_frames_to_process - 1, traj1[0].dx.shape[0], traj1[0].dx.shape[1]])
    for i in range(num_frames_to_process - 1):
        dx_trajectory1[i] = traj1[i].dx

    dx_trajectory2 = np.zeros([num_frames_to_process - 1, traj2[0].dx.shape[0], traj2[0].dx.shape[1]])
    for i in range(num_frames_to_process - 1):
        dx_trajectory2[i] = traj2[i].dx

    # Create a colormap based on time index
    colors = cm.viridis(np.linspace(0, 1, len(time)))

    # Create 3D scatter plots for the dx trajectory with a colormap
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(traj1[0].dx.shape[0]):
        x1 = dx_trajectory1[:, i, 0]
        y1 = dx_trajectory1[:, i, 1]
        z1 = dx_trajectory1[:, i, 2]
        ax1.scatter(x1, y1, z1, c=colors, cmap='viridis', s=10, marker='o', label='Atom {}'.format(i + 1))
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title('3D dx Trajectory with Time Index - Scale=1.0')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(traj2[0].dx.shape[0]):
        x2 = dx_trajectory2[:, i, 0]
        y2 = dx_trajectory2[:, i, 1]
        z2 = dx_trajectory2[:, i, 2]
        ax2.scatter(x2, y2, z2, c=colors, cmap='viridis', s=10, marker='^', label='Atom {}'.format(i + 1))
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_title('3D dx Trajectory with Time Index - Scale=2.0')
    ax2.legend()

    # Add colorbars to show the time index
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax1)
    cbar.set_label('Time Index')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
    cbar.set_label('Time Index')

    # Show the plot
    plt.show()

# Plot point distribution
def plt_point_distribution(num_frames_to_process, traj1, traj2, idx=0):
    # Generate coordinates over time
    time = np.arange(0, num_frames_to_process - 1)

    # point trajectory
    x1 = np.zeros(num_frames_to_process - 1)
    y1 = np.zeros(num_frames_to_process - 1)
    z1 = np.zeros(num_frames_to_process - 1)

    x2 = np.zeros(num_frames_to_process - 1)
    y2 = np.zeros(num_frames_to_process - 1)
    z2 = np.zeros(num_frames_to_process - 1)

    for i in range(num_frames_to_process - 1):
        x1[i] = traj1[i].dx[idx][0]
        y1[i] = traj1[i].dx[idx][1]
        z1[i] = traj1[i].dx[idx][2]
        x2[i] = traj2[i].dx[idx][0]
        y2[i] = traj2[i].dx[idx][1]
        z2[i] = traj2[i].dx[idx][2]

    # Create a colormap based on time index
    colors = cm.viridis(np.linspace(0, 1, len(time)))

    # Create a 3D scatter plot for the point trajectory with a colormap
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x1, y1, z1, c=colors, cmap='viridis', label='Trajectory')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title('3D Point Trajectory with Time Index - Scale=1.0')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x2, y2, z2, c=colors, cmap='viridis', label='Trajectory')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_title('3D Point Trajectory with Time Index - Scale=2.0')
    ax2.legend()

    # Add a colorbar to show the time index
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax1)
    cbar.set_label('Time Index')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
    cbar.set_label('Time Index')

    # Show the plot
    plt.show()

def augment_batch(batch):
    # Extract edge indices i, j from the batch
    i, j = batch.edge_index

    # Compute edge vectors (edge_vec) and edge lengths (edge_len)
    edge_vec = batch.pos[j] - batch.pos[i]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)

    # Concatenate edge vectors and edge lengths into edge_encoding
    batch.edge_encoding = torch.hstack([edge_vec, edge_len])
    return batch

def generate_loaders(dataset, parameters):
    pin_memory = parameters['pin_memory']
    num_workers = parameters['num_workers']
    batch_size = parameters['batch_size']
    train_size = parameters['train_size']

    # Train-validation split
    train_set, valid_set = train_test_split(dataset, train_size=train_size, random_state=42)
    print('Number of training graphs: {}'.format(len(train_set)))
    print('Number of validation graphs: {}'.format(len(valid_set)))

    # Move to data loaders
    kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': pin_memory, 'follow_batch': ['pos']}
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **kwargs)
    return batch_size, train_loader, valid_loader

# directory_path = '/home/she0000/PycharmProjects/pythonProject/Ex2/data/task2'
# cutoff = 1  # Angstrom
# scale = 1e-22
#
# TrajsDataset = TrajectoriesDataset(
#     directory_path,
#     cutoff=cutoff,
#     scale=scale
# )
# print('TrajsDataset[0].size: {}'.format(TrajsDataset[-1].size))
# node_dim = TrajsDataset[-1].x.shape[-1]
# print('init_node_dim: {}'.format(node_dim))
# config = parse_toml_file('../config.toml')
# data_dir = '/home/she0000/PycharmProjects/pythonProject/Ex2/data/'
# dataset_location = os.path.join(data_dir, 'dataset.pickle')
# pickle_object(TrajsDataset, dataset_location)
# dataset = unpickle_object(dataset_location)
# batch_size, train_loader, val_loader = generate_loaders(dataset, config)
# device0 = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# for batch in train_loader:
#     batch = augment_batch(batch)
#     batch_idx = batch.batch
#     edge_dim = batch.edge_encoding.shape[-1]
#     vector_dim = batch.pos.shape[-1]
#     break





