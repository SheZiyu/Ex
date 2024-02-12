import torch

from Ex2.model.ehgn_sparse import *
import torch.optim as optim

config = parse_toml_file('/home/she0000/PycharmProjects/pythonProject/Ex2/config.toml')

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
num_epochs = config['num_epochs']
num_splits = config['num_splits']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
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

# Initialize KFold for cross-validation
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

# Iterate over cross-validation splits
for fold, (train_indices, test_indices) in enumerate(kf.split(TrajsDataset.dataset)):
    # Set random seed for reproducibility in model initialization
    torch.manual_seed(42)

    # Initialize the model, optimizer, and criterion
    model = DynamicsEHGN_TopK_Sparse(node_dim, edge_dim, vector_dim, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Convert indices to PyTorch tensors
    train_indices = torch.from_numpy(train_indices)
    test_indices = torch.from_numpy(test_indices)

    # Create data loaders for training and testing
    train_loader = DataLoader(TrajsDataset, batch_size=batch_size, sampler=train_indices)
    test_loader = DataLoader(TrajsDataset, batch_size=batch_size, sampler=test_indices)

    # Reset the model parameters for the current fold
    model.reset_parameters()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{num_epochs} - Fold {fold + 1} - Training",
                          total=len(train_loader),
                          disable=False):
            batch = augment_batch(batch)
            batch_idx = batch.batch
            optimizer.zero_grad()
            _, outputs = model(
                t=batch.frame_idx,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_encoding,
                x=batch.pos,
                h=batch.x,
                batch_idx=batch_idx
            )
            loss = criterion(outputs.squeeze(1), batch.y.squeeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average training loss for the epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Fold {fold + 1} - Average Training Loss: {total_loss / len(train_loader)}")

        model.eval()
        total_loss_test = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader,
                              desc=f"Epoch {epoch + 1}/{num_epochs} - Fold {fold + 1} - Testing",
                              total=len(test_loader),
                              disable=False):
                batch = augment_batch(batch)
                batch_idx = batch.batch
                _, outputs = model(
                    t=batch.frame_idx,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_encoding,
                    x=batch.pos,
                    h=batch.x,
                    batch_idx=batch_idx
                )
                loss = criterion(outputs.squeeze(1), batch.y.squeeze(1))
                total_loss_test += loss.item()

        # Print average testing loss for the epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Fold {fold + 1} - Average Testing Loss: {total_loss_test / len(test_loader)}")








