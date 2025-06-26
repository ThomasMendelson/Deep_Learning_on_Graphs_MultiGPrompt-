#!/usr/bin/env python3
"""
VAE Decoder for MultiGPrompt
Reconstructs input features from embeddings produced by the PrePrompt encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAEDecoder(nn.Module):
    """
    VAE Decoder that reconstructs input features from embeddings
    
    Architecture:
    - Takes embeddings of dimension hid_units (256)
    - Reconstructs input features of dimension n_in (varies by dataset)
    - Uses a symmetric architecture to the encoder
    """
    
    def __init__(self, embedding_dim, feature_dim, hidden_dims=None, dropout=0.3):
        """
        Initialize VAE Decoder
        
        Args:
            embedding_dim (int): Dimension of input embeddings (hid_units = 256)
            feature_dim (int): Dimension of output features (n_in, varies by dataset)
            hidden_dims (list): List of hidden layer dimensions, defaults to [128, 64]
            dropout (float): Dropout rate for regularization
        """
        super(VAEDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        
        # Default hidden dimensions if not specified
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Build decoder layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final reconstruction layer
        layers.append(nn.Linear(input_dim, feature_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, embeddings):
        """
        Forward pass: reconstruct features from embeddings
        
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            torch.Tensor: Reconstructed features of shape (batch_size, feature_dim)
        """
        # Ensure embeddings are 2D
        if embeddings.dim() == 3:
            # Remove batch dimension if present (from PrePrompt.embed output)
            embeddings = embeddings.squeeze(0)
        
        # Reconstruct features
        reconstructed_features = self.decoder(embeddings)
        
        return reconstructed_features


class VAELoss(nn.Module):
    """
    VAE Loss combining reconstruction loss and KL divergence
    """
    
    def __init__(self, reconstruction_loss='mse', beta=1.0):
        """
        Initialize VAE Loss
        
        Args:
            reconstruction_loss (str): Type of reconstruction loss ('mse' or 'bce')
            beta (float): Weight for KL divergence term (beta-VAE)
        """
        super(VAELoss, self).__init__()
        
        self.beta = beta
        
        if reconstruction_loss == 'mse':
            self.reconstruction_loss = nn.MSELoss(reduction='mean')
        elif reconstruction_loss == 'bce':
            self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")
    
    def forward(self, reconstructed, original, mu=None, logvar=None):
        """
        Compute VAE loss
        
        Args:
            reconstructed (torch.Tensor): Reconstructed features
            original (torch.Tensor): Original features
            mu (torch.Tensor): Mean of latent distribution (for KL divergence)
            logvar (torch.Tensor): Log variance of latent distribution (for KL divergence)
            
        Returns:
            dict: Dictionary containing total loss, reconstruction loss, and KL loss
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, original)
        
        # KL divergence loss (if mu and logvar are provided)
        kl_loss = 0.0
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss.mean()
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }


class VAEModel(nn.Module):
    """
    Complete VAE model combining encoder (PrePrompt) and decoder
    """
    
    def __init__(self, encoder, decoder, feature_dim):
        """
        Initialize VAE Model
        
        Args:
            encoder: PrePrompt encoder model
            decoder: VAEDecoder model
            feature_dim (int): Dimension of input features
        """
        super(VAEModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.feature_dim = feature_dim
        
        # VAE components for latent space
        self.fc_mu = nn.Linear(encoder.gcn.convs[-1].fc.out_features, encoder.gcn.convs[-1].fc.out_features)
        self.fc_logvar = nn.Linear(encoder.gcn.convs[-1].fc.out_features, encoder.gcn.convs[-1].fc.out_features)
        
        # Initialize VAE components
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
    
    def encode(self, seq, adj, sparse, msk, LP):
        """
        Encode input features to latent space
        
        Args:
            seq: Input features
            adj: Adjacency matrix
            sparse: Whether to use sparse operations
            msk: Mask for readout
            LP: Whether to use LP mode
            
        Returns:
            tuple: (embeddings, mu, logvar)
        """
        # Get embeddings from encoder
        embeddings, _ = self.encoder.embed(seq, adj, sparse, msk, LP)
        
        # Get latent parameters
        mu = self.fc_mu(embeddings)
        logvar = self.fc_logvar(embeddings)
        
        return embeddings, mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vectors
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """
        Decode latent vectors to reconstructed features
        
        Args:
            z: Latent vectors
            
        Returns:
            torch.Tensor: Reconstructed features
        """
        return self.decoder(z)
    
    def forward(self, seq, adj, sparse, msk, LP):
        """
        Forward pass through the complete VAE
        
        Args:
            seq: Input features
            adj: Adjacency matrix
            sparse: Whether to use sparse operations
            msk: Mask for readout
            LP: Whether to use LP mode
            
        Returns:
            dict: Dictionary containing reconstructed features, mu, logvar, and embeddings
        """
        # Encode
        embeddings, mu, logvar = self.encode(seq, adj, sparse, msk, LP)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decode(z)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'embeddings': embeddings,
            'latent': z
        }


def create_vae_decoder(embedding_dim, feature_dim, hidden_dims=None, dropout=0.3):
    """
    Factory function to create a VAE decoder
    
    Args:
        embedding_dim (int): Dimension of input embeddings
        feature_dim (int): Dimension of output features
        hidden_dims (list): List of hidden layer dimensions
        dropout (float): Dropout rate
        
    Returns:
        VAEDecoder: Initialized VAE decoder
    """
    return VAEDecoder(embedding_dim, feature_dim, hidden_dims, dropout)


def create_vae_model(encoder, feature_dim, hidden_dims=None, dropout=0.3):
    """
    Factory function to create a complete VAE model
    
    Args:
        encoder: PrePrompt encoder model
        feature_dim (int): Dimension of input features
        hidden_dims (list): List of hidden layer dimensions for decoder
        dropout (float): Dropout rate
        
    Returns:
        VAEModel: Complete VAE model
    """
    # Get embedding dimension from encoder
    embedding_dim = encoder.gcn.convs[-1].fc.out_features
    
    # Create decoder
    decoder = create_vae_decoder(embedding_dim, feature_dim, hidden_dims, dropout)
    
    # Create complete VAE model
    vae_model = VAEModel(encoder, decoder, feature_dim)
    
    return vae_model


def train_vae_decoder(encoder, decoder, train_loader, val_loader, device, 
                     epochs=100, lr=0.001, beta=1.0, patience=20):
    """
    Train the VAE decoder
    
    Args:
        encoder: PrePrompt encoder model
        decoder: VAEDecoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs (int): Number of training epochs
        lr (float): Learning rate
        beta (float): Beta parameter for VAE loss
        patience (int): Early stopping patience
        
    Returns:
        dict: Training history
    """
    # Move models to device
    encoder.to(device)
    decoder.to(device)
    
    # Set encoder to evaluation mode (we don't train it)
    encoder.eval()
    
    # Set decoder to training mode
    decoder.train()
    
    # Loss function and optimizer
    criterion = VAELoss(reconstruction_loss='mse', beta=beta)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon_loss': [],
        'val_recon_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        train_recon_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            features, adj, mask, indices = batch
            features = features.to(device)
            adj = adj.to(device)
            mask = mask.to(device) if mask is not None else None
            
            # Get embeddings from encoder
            with torch.no_grad():
                embeddings, _ = encoder.embed(features, adj, True, mask, False)
            
            # FIXED: Handle the tensor dimensions correctly
            # The embeddings are [1, num_nodes, embedding_dim], we need [num_nodes, embedding_dim]
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)  # Remove batch dimension
            
            # Reconstruct features
            reconstructed = decoder(embeddings)
            
            # FIXED: Use the indices to get the correct target features
            # The embeddings correspond to the full graph, but we want to reconstruct
            # only the features for the specified indices
            if indices and len(indices) > 0:
                # Get the target features for the specified indices
                # Features are [1, num_nodes, feature_dim], we need [num_nodes, feature_dim]
                if features.dim() == 3:
                    features_flat = features.squeeze(0)  # Remove batch dimension
                else:
                    features_flat = features
                
                # Get target features for the specified indices
                target_features = features_flat[indices]
                
                # Get the corresponding embeddings for these indices
                embedding_subset = embeddings[indices]
                reconstructed = decoder(embedding_subset)
            else:
                # Use all features if no indices specified
                if features.dim() == 3:
                    target_features = features.squeeze(0)
                else:
                    target_features = features
            
            # Ensure reconstructed and target have the same shape
            if reconstructed.shape != target_features.shape:
                min_size = min(reconstructed.shape[0], target_features.shape[0])
                reconstructed = reconstructed[:min_size]
                target_features = target_features[:min_size]
            
            # Compute loss
            loss_dict = criterion(reconstructed, target_features)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_dict['reconstruction_loss'].item()
            num_batches += 1
        
        # Validation phase
        val_loss = 0.0
        val_recon_loss = 0.0
        num_val_batches = 0
        
        decoder.eval()
        with torch.no_grad():
            for batch in val_loader:
                features, adj, mask, indices = batch
                features = features.to(device)
                adj = adj.to(device)
                mask = mask.to(device) if mask is not None else None
                
                # Get embeddings from encoder
                embeddings, _ = encoder.embed(features, adj, True, mask, False)
                
                # FIXED: Handle the tensor dimensions correctly
                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(0)  # Remove batch dimension
                
                # Reconstruct features
                reconstructed = decoder(embeddings)
                
                # FIXED: Use the indices to get the correct target features for validation
                if indices and len(indices) > 0:
                    # Get the target features for the specified indices
                    if features.dim() == 3:
                        features_flat = features.squeeze(0)  # Remove batch dimension
                    else:
                        features_flat = features
                    
                    # Get target features for the specified indices
                    target_features = features_flat[indices]
                    
                    # Get the corresponding embeddings for these indices
                    embedding_subset = embeddings[indices]
                    reconstructed = decoder(embedding_subset)
                else:
                    if features.dim() == 3:
                        target_features = features.squeeze(0)
                    else:
                        target_features = features
                
                # Ensure reconstructed and target have the same shape
                if reconstructed.shape != target_features.shape:
                    min_size = min(reconstructed.shape[0], target_features.shape[0])
                    reconstructed = reconstructed[:min_size]
                    target_features = target_features[:min_size]
                
                # Compute loss
                loss_dict = criterion(reconstructed, target_features)
                
                val_loss += loss_dict['total_loss'].item()
                val_recon_loss += loss_dict['reconstruction_loss'].item()
                num_val_batches += 1
        
        decoder.train()
        
        # Average losses
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        avg_train_recon_loss = train_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_val_recon_loss = val_recon_loss / num_val_batches if num_val_batches > 0 else 0.0
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_recon_loss'].append(avg_train_recon_loss)
        history['val_recon_loss'].append(avg_val_recon_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Train Recon: {avg_train_recon_loss:.4f}, "
                  f"Val Recon: {avg_val_recon_loss:.4f}")
    
    return history


def evaluate_vae_reconstruction(encoder, decoder, test_loader, device):
    """
    Evaluate VAE reconstruction quality
    
    Args:
        encoder: PrePrompt encoder model
        decoder: VAEDecoder model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    encoder.eval()
    decoder.eval()
    
    total_recon_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            features, adj, mask, indices = batch
            features = features.to(device)
            adj = adj.to(device)
            mask = mask.to(device) if mask is not None else None
            
            # Get embeddings from encoder
            embeddings, _ = encoder.embed(features, adj, True, mask, False)
            
            # FIXED: Handle the tensor dimensions correctly
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)  # Remove batch dimension
            
            # Reconstruct features
            reconstructed = decoder(embeddings)
            
            # FIXED: Use the indices to get the correct target features for evaluation
            if indices and len(indices) > 0:
                # Get the target features for the specified indices
                if features.dim() == 3:
                    features_flat = features.squeeze(0)  # Remove batch dimension
                else:
                    features_flat = features
                
                # Get target features for the specified indices
                target_features = features_flat[indices]
                
                # Get the corresponding embeddings for these indices
                embedding_subset = embeddings[indices]
                reconstructed = decoder(embedding_subset)
            else:
                if features.dim() == 3:
                    target_features = features.squeeze(0)
                else:
                    target_features = features
            
            # Ensure reconstructed and target have the same shape
            if reconstructed.shape != target_features.shape:
                min_size = min(reconstructed.shape[0], target_features.shape[0])
                reconstructed = reconstructed[:min_size]
                target_features = target_features[:min_size]
            
            # Compute metrics
            mse_loss = F.mse_loss(reconstructed, target_features, reduction='mean')
            mae_loss = F.l1_loss(reconstructed, target_features, reduction='mean')
            
            total_recon_loss += mse_loss.item()
            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
            num_samples += 1
    
    # Average metrics
    avg_recon_loss = total_recon_loss / num_samples if num_samples > 0 else 0.0
    avg_mse = total_mse / num_samples if num_samples > 0 else 0.0
    avg_mae = total_mae / num_samples if num_samples > 0 else 0.0
    
    return {
        'reconstruction_loss': avg_recon_loss,
        'mse': avg_mse,
        'mae': avg_mae
    } 