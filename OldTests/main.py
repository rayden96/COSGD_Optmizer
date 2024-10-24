# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time

# Load and Prepare the Dataset
def load_or_create_dataset(filepath='dataset.csv', create_new=False, n_samples=1000, n_features=20):
    if create_new:
        data = np.random.rand(n_samples, n_features)
        labels = np.random.randint(0, 2, n_samples)
        dataset = pd.DataFrame(data)
        dataset['label'] = labels
        dataset.to_csv(filepath, index=False)
    else:
        dataset = pd.read_csv(filepath)
    
    return dataset

# Cluster the dataset
def cluster_dataset(dataset, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    features = dataset.drop('label', axis=1).values
    clusters = kmeans.fit_predict(features)
    dataset['cluster'] = clusters
    
    return dataset, clusters

# Define the model architecture
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(CustomModel, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Custom Optimizer COSGD
class COSGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(COSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        
        return loss

# Orthogonalize gradients using Gram-Schmidt process
def gram_schmidt(gradients):
    orthogonalized = []
    for g in gradients:
        w = g.clone()
        for og in orthogonalized:
            w -= torch.dot(w, og) * og
        w /= torch.norm(w)
        orthogonalized.append(w)
    return orthogonalized

# Train and Evaluate the Models
def train_model(model, optimizer, dataloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model

# Main Function to Execute the Pipeline
def main():
    # Load or create the dataset
    dataset = load_or_create_dataset(create_new=True)
    
    # Cluster the dataset
    dataset, clusters = cluster_dataset(dataset)
    
    # Prepare data loaders for each cluster
    dataloaders = {}
    for cluster in np.unique(clusters):
        cluster_data = dataset[dataset['cluster'] == cluster]
        features = cluster_data.drop(['label', 'cluster'], axis=1).values
        labels = cluster_data['label'].values
        tensor_data = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
        dataloaders[cluster] = DataLoader(tensor_data, batch_size=32, shuffle=True)
    
    # Define model architectures
    input_size = dataset.shape[1] - 2  # excluding label and cluster columns
    hidden_layers = [50, 20]
    output_size = 2
    
    model_sgd = CustomModel(input_size, hidden_layers, output_size)
    model_cosgd = CustomModel(input_size, hidden_layers, output_size)
    
    # Define optimizers
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
    optimizer_cosgd = COSGD(model_cosgd.parameters(), lr=0.01)
    
    # Train models
    start_time_sgd = time.time()
    for cluster, dataloader in dataloaders.items():
        train_model(model_sgd, optimizer_sgd, dataloader)
    end_time_sgd = time.time()
    
    start_time_cosgd = time.time()
    gradient_updates = []
    for cluster, dataloader in dataloaders.items():
        optimizer_cosgd.zero_grad()
        for inputs, labels in dataloader:
            outputs = model_cosgd(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
        gradients = [p.grad.data.clone() for p in model_cosgd.parameters()]
        gradient_updates.append(gradients)
    
    # Sort clusters by average gradient magnitude
    avg_gradients = [torch.mean(torch.stack([torch.norm(g) for g in grads])) for grads in gradient_updates]
    sorted_indices = np.argsort(avg_gradients)[::-1]
    sorted_gradients = [gradient_updates[i] for i in sorted_indices]
    
    # Orthogonalize gradients
    orthogonalized_gradients = [gram_schmidt(grads) for grads in sorted_gradients]
    
    # Apply orthogonalized gradients
    for grads in orthogonalized_gradients:
        for param, grad in zip(model_cosgd.parameters(), grads):
            param.data.add_(-optimizer_cosgd.param_groups[0]['lr'], grad)
    end_time_cosgd = time.time()
    
    # Print results
    print(f"SGD Training Time: {end_time_sgd - start_time_sgd:.2f} seconds")
    print(f"COSGD Training Time: {end_time_cosgd - start_time_cosgd:.2f} seconds")
    
    # Further analysis and evaluation can be done here
    
if __name__ == "__main__":
    main()
