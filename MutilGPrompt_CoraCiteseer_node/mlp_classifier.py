#!/usr/bin/env python3
"""
MLP Classifier Module
Contains the simple classifier implementation and all related functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from config import CONFIG, update_config

# Global variable to store the trained simple classifier
TRAINED_SIMPLE_CLASSIFIER = None
TRAINED_CLASSIFIER_EMBEDDINGS = None
TRAINED_CLASSIFIER_TRAIN_INDICES = None
TRAINING_COUNT = 0  # Track how many times we trained
REUSE_COUNT = 0     # Track how many times we reused

class SimpleClassifier(nn.Module):
    """
    Simple but effective classifier that takes 256-dimensional embeddings and outputs class predictions
    Features:
    - Simple architecture that works well
    - Batch normalization for stable training
    - ReLU activation for simplicity and effectiveness
    - Proper dropout for regularization
    - Good initialization
    """
    def __init__(self, input_dim=256, num_classes=7, hidden_dim=128, dropout=0.3):
        super(SimpleClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, embeddings):
        """
        Forward pass
        
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, 256)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        return self.classifier(embeddings)

def create_simple_classifier(num_classes, device='cuda', dropout=0.3, hidden_dim=128):
    """
    Factory function to create a simple but effective classifier
    
    Args:
        num_classes (int): Number of output classes
        device (str): Device to place the model on
        dropout (float): Dropout rate (default: 0.3)
        hidden_dim (int): Hidden dimension size (default: 128)
        
    Returns:
        SimpleClassifier: Initialized classifier model
    """
    model = SimpleClassifier(
        input_dim=256, 
        num_classes=num_classes, 
        hidden_dim=hidden_dim, 
        dropout=dropout
    )
    model.to(device)
    return model

def augment_embeddings(embeddings, strength=0.1, device='cuda'):
    """
    Apply data augmentation to embeddings
    
    Args:
        embeddings (torch.Tensor): Input embeddings
        strength (float): Augmentation strength
        device (str): Device to use
        
    Returns:
        torch.Tensor: Augmented embeddings
    """
    if not CONFIG['simple_classifier_use_data_augmentation']:
        return embeddings
    
    # Add Gaussian noise
    noise = torch.randn_like(embeddings) * strength
    augmented = embeddings + noise
    
    # Random scaling
    scale = torch.rand(embeddings.size(0), 1, device=device) * 0.2 + 0.9  # 0.9 to 1.1
    augmented = augmented * scale
    
    return augmented

def mixup_data(embeddings, labels, alpha=0.2, device='cuda'):
    """
    Apply mixup data augmentation
    
    Args:
        embeddings (torch.Tensor): Input embeddings
        labels (torch.Tensor): Input labels (one-hot)
        alpha (float): Mixup alpha parameter
        device (str): Device to use
        
    Returns:
        tuple: (mixed_embeddings, original_labels, lam)
    """
    if not CONFIG['simple_classifier_use_mixup']:
        return embeddings, labels, 1.0
    
    batch_size = embeddings.size(0)
    
    # Generate mixing weights
    lam = np.random.beta(alpha, alpha)
    
    # Shuffle indices
    index = torch.randperm(batch_size).to(device)
    
    # Mix embeddings only, keep original labels for CrossEntropyLoss
    mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
    
    return mixed_embeddings, labels, lam

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Print training summary instead of plotting curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Ignored (kept for compatibility)
    """
    if len(train_losses) > 0:
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        
        # Show best validation accuracy
        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1
        print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%) at epoch {best_epoch}")

def plot_enhanced_training_curves(train_losses, val_losses, train_accs, val_accs, learning_rates=None, save_path=None):
    """
    Print enhanced training summary instead of plotting curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        learning_rates: List of learning rates per epoch (ignored)
        save_path: Ignored (kept for compatibility)
    """
    if len(train_losses) > 0:
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        
        # Show best validation accuracy
        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1
        print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%) at epoch {best_epoch}")
        
        # Show convergence info
        if len(train_losses) > 20:
            recent_train_loss = train_losses[-10:]
            recent_val_loss = val_losses[-10:]
            train_loss_std = np.std(recent_train_loss)
            val_loss_std = np.std(recent_val_loss)
            print(f"Recent Training Loss Stability: std={train_loss_std:.4f}")
            print(f"Recent Validation Loss Stability: std={val_loss_std:.4f}")

def calculate_batch_info(train_samples, val_samples, batch_size):
    """
    Calculate and display batch information for training and validation
    
    Args:
        train_samples (int): Number of training samples
        val_samples (int): Number of validation samples
        batch_size (int): Batch size
    """
    train_batches = (train_samples + batch_size - 1) // batch_size  # Ceiling division
    val_batches = 1  # Validation uses full set as single batch
    
    print(f"Training: {train_samples} samples → {train_batches} batches")
    print(f"Validation: {val_samples} samples → 1 batch (full set)")
    
    # Calculate effective batch sizes (last batch might be smaller)
    train_last_batch_size = train_samples % batch_size
    if train_last_batch_size > 0:
        print(f"Note: Last training batch will have {train_last_batch_size} samples")
    print(f"Note: Validation uses all {val_samples} samples in one batch")

def train_simple_classifier(model, train_embeddings, train_labels, val_embeddings, val_labels, 
                          lr=0.001, epochs=2000, patience=100, dropout=0.2, batch_size=32):
    """
    Train the improved simple classifier with advanced training techniques
    
    Args:
        model: The classifier model to train
        train_embeddings (torch.Tensor): Training embeddings
        train_labels (torch.Tensor): Training labels (one-hot encoded)
        val_embeddings (torch.Tensor): Validation embeddings
        val_labels (torch.Tensor): Validation labels (one-hot encoded)
        lr (float): Learning rate
        epochs (int): Number of training epochs
        patience (int): Early stopping patience (epochs without improvement)
        dropout (float): Dropout rate
        batch_size (int): Batch size for training
        
    Returns:
        trained model
    """
    device = next(model.parameters()).device
    model.train()
    
    # Convert to tensors and move to device
    train_embeddings = torch.FloatTensor(train_embeddings).to(device)
    train_labels = torch.FloatTensor(train_labels).to(device)
    val_embeddings = torch.FloatTensor(val_embeddings).to(device)
    val_labels = torch.FloatTensor(val_labels).to(device)
    
    # Create data loaders for batch training
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Use full validation set as single batch
    val_dataset = TensorDataset(val_embeddings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=len(val_embeddings), shuffle=False)
    
    # Calculate and display batch information
    calculate_batch_info(len(train_embeddings), len(val_embeddings), batch_size)
    
    print(f"Training with batch size: {batch_size}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation: Full set ({len(val_embeddings)} samples) in 1 batch")
    
    # Setup optimizer with improved configuration
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=CONFIG['simple_classifier_weight_decay']
    )
    
    # Setup learning rate scheduler (disabled for simplicity)
    scheduler = None
    print("No learning rate scheduler (simplified training)")
    
    # Setup loss function (no label smoothing for simplicity)
    criterion = nn.CrossEntropyLoss()
    
    # Helper function to convert one-hot to class indices
    def one_hot_to_indices(one_hot_labels):
        return torch.argmax(one_hot_labels, dim=1)
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # History tracking for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # learning_rates = []  # Removed since we're not using a scheduler
    
    print(f"Starting simplified training for {epochs} epochs with patience {patience}...")
    print(f"Training data shape: {train_embeddings.shape}")
    print(f"Validation data shape: {val_embeddings.shape}")
    print(f"Number of classes: {train_labels.shape[1]}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    
    # Debug: Check model initialization
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters - Total: {total_params}, Trainable: {trainable_params}")
    
    # Check if any parameters are all zeros (bad initialization)
    zero_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad and torch.all(param == 0):
            zero_params += 1
    
    if zero_params == 0:
        print("All trainable parameters are properly initialized")
    else:
        print(f"{zero_params} parameters are all zeros")
    
    # Test model forward pass
    model.eval()
    with torch.no_grad():
        test_output = model(train_embeddings[:1])  # Test with one sample
        print(f"Test output shape: {test_output.shape}")
    
    model.train()  # Set back to training mode
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_embeddings, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass (no augmentation for simplicity)
            outputs = model(batch_embeddings)
            
            # Calculate loss
            # Convert one-hot labels to class indices for CrossEntropyLoss
            batch_labels_indices = one_hot_to_indices(batch_labels)
            train_loss = criterion(outputs, batch_labels_indices)
            
            # Backward pass
            train_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate training accuracy for this batch
            train_pred = torch.argmax(outputs, dim=1)
            train_true = torch.argmax(batch_labels, dim=1)
            epoch_train_correct += (train_pred == train_true).sum().item()
            epoch_train_total += batch_labels.size(0)
            epoch_train_loss += train_loss.item()
        
        # Calculate epoch training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_correct / epoch_train_total
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        
        with torch.no_grad():
            # Validation is now a single batch with all validation data
            for batch_embeddings, batch_labels in val_loader:
                val_outputs = model(batch_embeddings)
                # Convert one-hot labels to class indices for loss calculation
                batch_labels_indices = one_hot_to_indices(batch_labels)
                val_loss = criterion(val_outputs, batch_labels_indices)
                
                val_pred = torch.argmax(val_outputs, dim=1)
                val_true = torch.argmax(batch_labels, dim=1)
                epoch_val_correct += (val_pred == val_true).sum().item()
                epoch_val_total += batch_labels.size(0)
                epoch_val_loss += val_loss.item()
        
        # Calculate epoch validation metrics (single batch, so no averaging needed)
        avg_val_loss = epoch_val_loss  # Single batch, so this is already the average
        avg_val_acc = epoch_val_correct / epoch_val_total
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            # learning_rates.append(current_lr)
        # else:
        #     learning_rates.append(lr)
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)
        
        # Early stopping check (use validation accuracy instead of loss for better stability)
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, "
                  f"LR: {lr:.2e}, Patience: {patience_counter}/{patience}")
            
            # Debug: Check if model is learning
            if epoch == 0:
                print(f"Debug: First epoch outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"Debug: First epoch predictions: {train_pred[:5].cpu().numpy()}")
                print(f"Debug: First epoch true labels: {train_true[:5].cpu().numpy()}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    # Print training summary
    print("\nTRAINING COMPLETED")
    print("="*40)
    plot_enhanced_training_curves(train_losses, val_losses, train_accs, val_accs, 
                                 learning_rates=None, save_path=None)
    
    return model

def evaluate_simple_classifier(model, test_embeddings, test_labels, device='cuda'):
    """
    Evaluate the simple classifier
    
    Args:
        model (SimpleClassifier): The trained classifier model
        test_embeddings (torch.Tensor): Test embeddings
        test_labels (torch.Tensor): Test labels (one-hot encoded)
        device (str): Device to use for evaluation
        
    Returns:
        tuple: (predictions, true_labels, probabilities, logits)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(test_embeddings)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        
        # Convert one-hot labels to class indices
        true_labels = torch.argmax(test_labels, dim=1)
        
        return (predictions.cpu().numpy(), 
                true_labels.cpu().numpy(), 
                probabilities.cpu().numpy(), 
                outputs.cpu().numpy())

def evaluate_simple_classifier_detailed(model, test_embeddings, test_labels, device='cuda'):
    """
    Enhanced evaluation of the simple classifier with detailed metrics
    
    Args:
        model (SimpleClassifier): The trained classifier model
        test_embeddings (torch.Tensor): Test embeddings
        test_labels (torch.Tensor): Test labels (one-hot encoded)
        device (str): Device to use for evaluation
        
    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        outputs = model(test_embeddings)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        
        # Convert one-hot labels to class indices
        true_labels = torch.argmax(test_labels, dim=1)
        
        # Calculate detailed metrics
        accuracy = (predictions == true_labels).float().mean().item()
        
        # Calculate confidence metrics
        max_probs = torch.max(probabilities, dim=1)[0]
        avg_confidence = max_probs.mean().item()
        confidence_std = max_probs.std().item()
        
        # Calculate entropy of predictions (measure of uncertainty)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        avg_entropy = entropy.mean().item()
        
        # Calculate per-class accuracy
        num_classes = test_labels.shape[1]
        per_class_accuracy = []
        for i in range(num_classes):
            class_mask = (true_labels == i)
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == true_labels[class_mask]).float().mean().item()
                per_class_accuracy.append(class_acc)
            else:
                per_class_accuracy.append(0.0)
        
        # Calculate F1 score
        f1_macro = f1_score(true_labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        f1_weighted = f1_score(true_labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
        
        results = {
            'predictions': predictions.cpu().numpy(),
            'true_labels': true_labels.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'logits': outputs.cpu().numpy(),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'avg_entropy': avg_entropy,
            'per_class_accuracy': per_class_accuracy,
            'num_classes': num_classes
        }
        
        return results

def analyze_confidence_calibration(predictions, true_labels, probabilities, num_bins=10):
    """
    Analyze the confidence calibration of the model predictions
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        probabilities: Prediction probabilities
        num_bins: Number of confidence bins
        
    Returns:
        dict: Calibration analysis results
    """
    # Calculate confidence for each prediction
    confidences = np.max(probabilities, axis=1)
    
    # Create confidence bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate calibration metrics
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()
        
        if bin_count > 0:
            bin_accuracy = (predictions[in_bin] == true_labels[in_bin]).mean()
            bin_confidence = confidences[in_bin].mean()
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    # Calculate Expected Calibration Error (ECE)
    ece = 0.0
    for i in range(len(bin_accuracies)):
        if bin_counts[i] > 0:
            ece += bin_counts[i] * abs(bin_accuracies[i] - bin_confidences[i])
    ece /= len(predictions)
    
    # Calculate reliability diagram
    reliability_diag = {
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'ece': ece,
        'num_bins': num_bins
    }
    
    return reliability_diag

def plot_calibration_analysis(reliability_diag, save_path=None):
    """
    Print confidence calibration analysis instead of plotting
    
    Args:
        reliability_diag: Results from analyze_confidence_calibration
        save_path: Ignored (kept for compatibility)
    """
    bin_accuracies = reliability_diag['bin_accuracies']
    bin_confidences = reliability_diag['bin_confidences']
    bin_counts = reliability_diag['bin_counts']
    ece = reliability_diag['ece']
    
    print("\nCONFIDENCE CALIBRATION ANALYSIS:")
    print("="*50)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    print("\nReliability Table:")
    print("Confidence | Accuracy | Count")
    print("-" * 30)
    
    for i in range(len(bin_accuracies)):
        if bin_counts[i] > 0:
            print(f"{bin_confidences[i]:9.3f} | {bin_accuracies[i]:8.3f} | {bin_counts[i]:5d}")
    
    print("-" * 30)
    
    # Calibration quality assessment
    if ece < 0.05:
        calibration_quality = "Excellent"
    elif ece < 0.1:
        calibration_quality = "Good"
    elif ece < 0.2:
        calibration_quality = "Fair"
    else:
        calibration_quality = "Poor"
    
    print(f"Calibration Quality: {calibration_quality}")
    
    # Show distribution of confidence scores
    total_predictions = sum(bin_counts)
    if total_predictions > 0:
        print(f"\nConfidence Distribution:")
        for i in range(len(bin_counts)):
            if bin_counts[i] > 0:
                percentage = (bin_counts[i] / total_predictions) * 100
                print(f"  {bin_confidences[i]:.2f}-{bin_confidences[i]+0.1:.2f}: {bin_counts[i]} ({percentage:.1f}%)")
    
    print("="*50)

def reset_simple_classifier():
    """Reset the global simple classifier to force retraining"""
    global TRAINED_SIMPLE_CLASSIFIER, TRAINED_CLASSIFIER_EMBEDDINGS, TRAINED_CLASSIFIER_TRAIN_INDICES, TRAINING_COUNT, REUSE_COUNT
    TRAINED_SIMPLE_CLASSIFIER = None
    TRAINED_CLASSIFIER_EMBEDDINGS = None
    TRAINED_CLASSIFIER_TRAIN_INDICES = None
    TRAINING_COUNT = 0
    REUSE_COUNT = 0
    print("Simple classifier reset - will train new model on next evaluation")

def configure_simple_classifier(**kwargs):
    """
    Configure simple classifier parameters
    
    Args:
        **kwargs: Configuration parameters to update
                  Examples:
                  - epochs=1000
                  - lr=0.001
                  - dropout=0.2
                  - batch_size=64
                  - hidden_dim=256
                  - patience=100
    """
    # Map common parameter names to config keys
    param_mapping = {
        'epochs': 'simple_classifier_epochs',
        'lr': 'simple_classifier_lr',
        'dropout': 'simple_classifier_dropout',
        'batch_size': 'simple_classifier_batch_size',
        'hidden_dim': 'simple_classifier_hidden_dim',
        'patience': 'simple_classifier_patience',
        'weight_decay': 'simple_classifier_weight_decay',
        'train_val_split': 'train_val_split_ratio'
    }
    
    # Convert parameters to config keys
    config_updates = {}
    for key, value in kwargs.items():
        if key in param_mapping:
            config_updates[param_mapping[key]] = value
        else:
            # Assume it's already a config key
            config_updates[key] = value
    
    # Update configuration
    update_config(**config_updates)
    
    print("Simple classifier configuration updated:")
    for key, value in config_updates.items():
        print(f"   {key}: {value}")
    
    # Reset classifier to use new configuration
    reset_simple_classifier()

def print_simple_classifier_summary():
    """
    Print a comprehensive summary of the simplified simple classifier features
    """
    print("\n" + "="*60)
    print("SIMPLIFIED SIMPLE CLASSIFIER SUMMARY")
    print("="*60)
    
    print("\nHYPERPARAMETERS:")
    print(f"   ✅ Learning rate: {CONFIG['simple_classifier_lr']} (higher for faster convergence)")
    print(f"   ✅ Epochs: {CONFIG['simple_classifier_epochs']} (reduced for faster training)")
    print(f"   ✅ Patience: {CONFIG['simple_classifier_patience']} (reasonable early stopping)")
    print(f"   ✅ Dropout: {CONFIG['simple_classifier_dropout']} (standard regularization)")
    print(f"   ✅ Batch size: {CONFIG['simple_classifier_batch_size']} (standard size)")
    print(f"   ✅ Hidden dimension: {CONFIG['simple_classifier_hidden_dim']} (smaller, manageable)")
    print(f"   ✅ Weight decay: {CONFIG['simple_classifier_weight_decay']} (L2 regularization)")
    