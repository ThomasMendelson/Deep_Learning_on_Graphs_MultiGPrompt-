#!/usr/bin/env python3
"""
Configuration file for the simplified downprompt model testing script
"""

import torch

CONFIG = {
    # Model configuration
    'hid_units': 256,
    'nonlinearity': 'prelu',
    'sparse': True,
    'drop_percent': 0.1,
    
    # Evaluation configuration
    'sample_size': 100,
    'use_all_test_data': False,
    'single_run_mode': False,
    
    # Simple classifier configuration
    'use_simple_classifier': True,
    'simple_classifier_epochs': 500,
    'simple_classifier_lr': 0.01,
    'simple_classifier_dropout': 0.3,
    'simple_classifier_patience': 50,
    'simple_classifier_batch_size': 32,
    'train_val_split_ratio': 0.8,
    'simple_classifier_hidden_dim': 128,
    'simple_classifier_weight_decay': 1e-4,
    'simple_classifier_use_lr_scheduler': False,
    'simple_classifier_use_data_augmentation': False,
    'simple_classifier_augmentation_strength': 0.1,
    'simple_classifier_use_mixup': False,
    'simple_classifier_mixup_alpha': 0.2,
    
    # VAE Decoder configuration
    'use_vae_decoder': False,
    'vae_decoder_epochs': 100,
    'vae_decoder_lr': 0.001,
    'vae_decoder_beta': 1.0,
    'vae_decoder_dropout': 0.3,
    'vae_decoder_hidden_dims': [128, 64],
    'vae_decoder_patience': 20,
    'vae_decoder_batch_size': 1,
    'vae_decoder_reconstruction_loss': 'mse',
    'vae_decoder_train_on_train_only': True,
    'vae_decoder_evaluate_reconstruction': True,
    
    # Model paths
    'model_paths': {
        'cora': '../modelset/cora.pkl',
        'citeseer': '../modelset/citeseer.pkl'
    },
    
    # Device configuration
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Dataset configuration
    'dataset': 'cora',
    
    # Hyperparameters
    'a1': 0.9,
    'a2': 0.9,
    'a3': 0.1,
    'a4': 0.0001,
    
    # Random seed
    'seed': 36,
    
    # Run tracking
    'run_id': 1
}

def update_config(**kwargs):
    """Update configuration with new values"""
    CONFIG.update(kwargs)

def get_config():
    """Get the current configuration"""
    return CONFIG.copy()

def print_config():
    """Print the current configuration"""
    print("📋 CURRENT CONFIGURATION:")
    print("="*50)
    for key, value in CONFIG.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    print("="*50) 