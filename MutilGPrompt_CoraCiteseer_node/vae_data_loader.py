#!/usr/bin/env python3
"""
Data Loader for VAE Decoder Training
Handles graph data loading and batching for VAE training
"""

import torch
import torch.utils.data as data
import numpy as np
import scipy.sparse as sp
from utils import process


class GraphDataset(data.Dataset):
    """
    Dataset class for graph data used in VAE training
    """
    
    def __init__(self, features, adj, mask=None, indices=None):
        """
        Initialize GraphDataset
        
        Args:
            features: Node features
            adj: Adjacency matrix
            mask: Optional mask for readout
            indices: Optional indices to subset the data
        """
        self.features = features
        self.adj = adj
        self.mask = mask
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(features.shape[0]))
    
    def __len__(self):
        return 1  # Always return 1 since we work with the full graph
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Index of the sample (ignored, always returns full graph)
            
        Returns:
            tuple: (features, adj, mask, indices)
        """
        # Return the full graph data along with the indices
        return self.features, self.adj, self.mask, self.indices


class GraphDataLoader:
    """
    Data loader for graph data with batching support
    """
    
    def __init__(self, features, adj, mask=None, batch_size=1, shuffle=True, indices=None):
        """
        Initialize GraphDataLoader
        
        Args:
            features: Node features
            adj: Adjacency matrix
            mask: Optional mask for readout
            batch_size: Batch size (default 1 for full graph)
            shuffle: Whether to shuffle data
            indices: Optional indices to subset the data
        """
        self.dataset = GraphDataset(features, adj, mask, indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def get_loader(self):
        """
        Get PyTorch DataLoader
        
        Returns:
            torch.utils.data.DataLoader: Data loader
        """
        return data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """
        Custom collate function for graph data
        
        Args:
            batch: List of (features, adj, mask, indices) tuples
            
        Returns:
            tuple: Batched (features, adj, mask, indices)
        """
        # For graph data, we typically return the full graph
        # since batching graphs is complex
        features, adj, mask, indices = batch[0]
        return features, adj, mask, indices


def create_vae_data_loaders(features, adj, idx_train, idx_val, idx_test, 
                           batch_size=1, shuffle=True):
    """
    Create train, validation, and test data loaders for VAE training
    
    Args:
        features: Node features
        adj: Adjacency matrix
        idx_train: Training indices
        idx_val: Validation indices
        idx_test: Test indices
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create train loader (use train + val for training)
    train_indices = list(idx_train) + list(idx_val)
    train_loader = GraphDataLoader(
        features, adj, batch_size=batch_size, 
        shuffle=shuffle, indices=train_indices
    ).get_loader()
    
    # Create validation loader (use a subset of train for validation)
    val_size = min(len(idx_train) // 5, 100)  # Use 20% of train or max 100 samples
    val_indices = list(idx_train[:val_size])
    val_loader = GraphDataLoader(
        features, adj, batch_size=batch_size,
        shuffle=False, indices=val_indices
    ).get_loader()
    
    # Create test loader (only if test indices are provided)
    if len(idx_test) > 0:
        test_loader = GraphDataLoader(
            features, adj, batch_size=batch_size,
            shuffle=False, indices=list(idx_test)
        ).get_loader()
    else:
        # Create a dummy test loader with some training data
        test_indices = list(idx_train[:min(50, len(idx_train))])
        test_loader = GraphDataLoader(
            features, adj, batch_size=batch_size,
            shuffle=False, indices=test_indices
        ).get_loader()
    
    return train_loader, val_loader, test_loader


def prepare_vae_training_data(adj, features, labels, idx_train, idx_val, idx_test):
    """
    Prepare data for VAE training
    
    Args:
        adj: Adjacency matrix
        features: Node features
        labels: Node labels
        idx_train: Training indices
        idx_val: Validation indices
        idx_test: Test indices
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_dim)
    """
    # Convert features to tensor if needed
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)
    
    # Convert adj to sparse tensor if needed
    if not isinstance(adj, torch.Tensor):
        adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    
    # Get feature dimension
    feature_dim = features.shape[1]
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_vae_data_loaders(
        features, adj, idx_train, idx_val, idx_test
    )
    
    return train_loader, val_loader, test_loader, feature_dim


def create_subgraph_data_loaders(features, adj, idx_train, idx_val, idx_test,
                                subgraph_size=100, batch_size=1):
    """
    Create data loaders with subgraph sampling for more efficient training
    
    Args:
        features: Node features
        adj: Adjacency matrix
        idx_train: Training indices
        idx_val: Validation indices
        idx_test: Test indices
        subgraph_size: Size of subgraphs to sample
        batch_size: Batch size
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_dim)
    """
    # This is a simplified version - in practice you'd want more sophisticated subgraph sampling
    feature_dim = features.shape[1]
    
    # For now, use the simple approach
    return create_vae_data_loaders(features, adj, idx_train, idx_val, idx_test, batch_size) 