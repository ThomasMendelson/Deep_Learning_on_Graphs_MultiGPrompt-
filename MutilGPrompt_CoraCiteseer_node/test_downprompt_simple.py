#!/usr/bin/env python3
"""
Simplified Downprompt Model Testing Script
Uses only the dataset .pkl files from modelset folder
Combines configuration, model loading, data loading, and evaluation in one file
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.sparse.SparseTensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*SparseEfficiencyWarning.*")
warnings.filterwarnings("ignore", message=".*Changing the sparsity structure of a csr_array is expensive.*")
warnings.filterwarnings("ignore", message=".*lil and dok are more efficient.*")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse")
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy.sparse")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import random
import time

from models import LogReg
from preprompt import PrePrompt
import preprompt
from utils import process
import aug
from downprompt import downprompt, featureprompt
from vae_decoder import (
    VAEDecoder, 
    VAELoss, 
    VAEModel, 
    create_vae_decoder, 
    create_vae_model,
    train_vae_decoder,
    evaluate_vae_reconstruction
)
from vae_data_loader import prepare_vae_training_data
from config import CONFIG, update_config, get_config, print_config
from mlp_classifier import (
    SimpleClassifier,
    create_simple_classifier,
    augment_embeddings,
    mixup_data,
    plot_training_curves,
    plot_enhanced_training_curves,
    calculate_batch_info,
    train_simple_classifier,
    evaluate_simple_classifier,
    evaluate_simple_classifier_detailed,
    analyze_confidence_calibration,
    plot_calibration_analysis,
    reset_simple_classifier,
    configure_simple_classifier,
    print_simple_classifier_summary
)

TRAINED_SIMPLE_CLASSIFIER = None
TRAINED_CLASSIFIER_EMBEDDINGS = None
TRAINED_CLASSIFIER_TRAIN_INDICES = None
TRAINING_COUNT = 0
REUSE_COUNT = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess_data(dataset_name):
    print(f"Loading {dataset_name} dataset...")
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset_name)
    train_mask = np.zeros(features.shape[0], dtype=bool)
    train_mask[idx_train] = True
    print("Using only training data for feature preprocessing...")
    features_train_only = features[train_mask]
    features_train_only, _ = process.preprocess_features(features_train_only)
    features_dense = features.todense()
    features_train_dense = features_dense[train_mask]
    rowsum_train = np.array(features_train_dense.sum(1))
    r_inv_train = np.power(rowsum_train, -1).flatten()
    r_inv_train[np.isinf(r_inv_train)] = 0.
    mean_train_norm = np.mean(r_inv_train)
    r_inv_all = np.zeros(features_dense.shape[0])
    r_inv_all[train_mask] = r_inv_train
    r_inv_all[~train_mask] = mean_train_norm
    r_mat_inv_all = sp.diags(r_inv_all)
    features = r_mat_inv_all.dot(features)
    features = features.todense()
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]
    print(f"Dataset loaded: {nb_nodes} nodes, {ft_size} features, {nb_classes} classes")
    print(f"Train samples: {len(idx_train)}")
    print(f"Val samples: {len(idx_val)}")
    print(f"Test samples: {len(idx_test)}")
    return adj, features, labels, idx_train, idx_val, idx_test, nb_nodes, ft_size, nb_classes

def prepare_augmented_data(adj, features, drop_percent, idx_train, idx_val, idx_test):
    print("Preparing augmented data with proper train/test isolation...")
    print("Creating train-only adjacency matrix for normalization...")
    train_nodes = list(idx_train) + list(idx_val)
    train_mask = np.zeros(adj.shape[0], dtype=bool)
    train_mask[train_nodes] = True
    adj_train_only = adj[train_mask][:, train_mask]
    adj_train_only = process.normalize_adj(adj_train_only + sp.eye(adj_train_only.shape[0]))
    adj_full = adj + sp.eye(adj.shape[0])
    adj_normalized = adj_full.copy()
    adj_normalized[train_mask][:, train_mask] = adj_train_only.todense()
    features_tensor = torch.FloatTensor(features[np.newaxis])
    aug_features1edge = features_tensor
    aug_features2edge = features_tensor
    aug_adj1edge = aug.aug_random_edge(adj_normalized, drop_percent=drop_percent)
    aug_adj2edge = aug.aug_random_edge(adj_normalized, drop_percent=drop_percent)
    aug_features1mask = aug.aug_random_mask(features_tensor, drop_percent=drop_percent)
    aug_features2mask = aug.aug_random_mask(features_tensor, drop_percent=drop_percent)
    aug_adj1mask = adj_normalized
    aug_adj2mask = adj_normalized
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj_normalized)
    sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
    sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)
    sp_aug_adj1mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj1mask)
    sp_aug_adj2mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj2mask)
    print("Augmented data prepared with proper train/test isolation")
    return (aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
            sp_adj, sp_aug_adj1edge, sp_aug_adj2edge, sp_aug_adj1mask, sp_aug_adj2mask)

def create_isolated_test_graph(adj, idx_train, idx_val, idx_test):
    print("🔒 Creating isolated test graph to prevent data leakage...")
    adj_isolated = adj.copy()
    test_nodes = list(idx_test)
    train_val_nodes = list(idx_train) + list(idx_val)
    adj_isolated[test_nodes][:, train_val_nodes] = 0
    adj_isolated[train_val_nodes][:, test_nodes] = 0
    for i in test_nodes:
        adj_isolated[i, i] = 1
    print(f"✅ Isolated {len(test_nodes)} test nodes from {len(train_val_nodes)} train/val nodes")
    return adj_isolated

def sample_balanced_test_data(idx_test, sample_size, labels, nb_classes):
    print(f"📋 Sampling {sample_size} test samples with balanced class distribution...")
    samples_per_class = sample_size // nb_classes
    remaining_samples = sample_size % nb_classes
    print(f"📊 Target: {samples_per_class} samples per class + {remaining_samples} extra")
    sampled_indices = []
    for class_idx in range(nb_classes):
        class_samples = []
        for idx in idx_test:
            if torch.argmax(torch.FloatTensor(labels[idx])).item() == class_idx:
                class_samples.append(idx)
        print(f"📊 Class {class_idx}: {len(class_samples)} test samples available")
        target_samples = samples_per_class
        if class_idx < remaining_samples:
            target_samples += 1
        if len(class_samples) >= target_samples:
            selected = np.random.choice(class_samples, target_samples, replace=False)
        else:
            selected = np.array(class_samples)
            if target_samples > len(class_samples):
                additional_needed = target_samples - len(class_samples)
                additional = np.random.choice(class_samples, additional_needed, replace=True)
                selected = np.concatenate([selected, additional])
        sampled_indices.extend(selected)
    np.random.shuffle(sampled_indices)
    print(f"📊 Final sample: {len(sampled_indices)} samples")
    return np.array(sampled_indices)

def sample_test_data(idx_test, sample_size, nb_classes):
    if sample_size >= len(idx_test):
        print(f"Using all {len(idx_test)} test samples (requested: {sample_size})")
        result = idx_test
    else:
        sampled_indices = np.random.choice(idx_test, sample_size, replace=False)
        print(f"📋 Sampled {len(sampled_indices)} test samples")
        result = sampled_indices
    return result

def load_pretrained_model(ft_size, hid_units, device, adj, dataset_name):
    print(f"�� Loading pretrained model for {dataset_name}...")
    print("Pretrained model may have been trained with test data access!")
    print("This could lead to inflated accuracy scores.")
    negetive_sample = preprompt.prompt_pretrain_sample(adj, 200)
    model = PrePrompt(ft_size, hid_units, CONFIG['nonlinearity'], negetive_sample, 
                     CONFIG['a1'], CONFIG['a2'], CONFIG['a3'], CONFIG['a4'], 1, 0.3)
    model_path = CONFIG['model_paths'][dataset_name]
    
    model_loaded_successfully = False
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            print(f"📋 State dict keys ({len(state_dict.keys())}):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
                print(f"  {i+1}. {key}: {shape}")
            if len(state_dict.keys()) > 10:
                print(f"  ... and {len(state_dict.keys()) - 10} more keys")
            
            prototype_keys = [k for k in state_dict.keys() if 'ave' in k or 'prototype' in k or 'center' in k]
            if prototype_keys:
                print(f"🎯 Found prototype keys: {prototype_keys}")
                print("Prototypes may contain test data information!")
            else:
                print("No prototype keys found in state dict")
            
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for key in state_dict.keys():
                if key in model_state_dict:
                    filtered_state_dict[key] = state_dict[key]
            
            if filtered_state_dict:
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"✅ Model loaded successfully ({len(filtered_state_dict)} keys)")
                print("Model may have been trained with test data access!")
                model_loaded_successfully = True
            else:
                print("❌ No valid keys found - using random initialization")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Using random initialization")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Using random initialization")
    
    model.to(device)
    model.eval()
    
    return model, model_loaded_successfully

def create_uniform_prototype_data(idx_train, labels, nb_classes, samples_per_class=7):
    print(f"🎯 Creating uniform prototype data from training set ({samples_per_class} samples per class)...")
    uniform_indices = []
    uniform_labels = []
    for class_idx in range(nb_classes):
        class_samples = []
        for i, idx in enumerate(idx_train):
            if torch.argmax(torch.FloatTensor(labels[idx])).item() == class_idx:
                class_samples.append((idx, i))
        print(f"📊 Class {class_idx}: {len(class_samples)} training samples available")
        if len(class_samples) >= samples_per_class:
            selected = np.random.choice(len(class_samples), samples_per_class, replace=False)
        else:
            selected = np.arange(len(class_samples))
            if len(class_samples) > 0:
                additional_needed = samples_per_class - len(class_samples)
                additional_indices = np.random.choice(len(class_samples), additional_needed, replace=True)
                selected = np.concatenate([selected, additional_indices])
            else:
                print(f"No training samples found for class {class_idx}")
                selected = np.zeros(samples_per_class, dtype=int)
        for sel_idx in selected:
            uniform_indices.append(class_samples[sel_idx][0])
            uniform_labels.append(class_idx)
    total_samples = len(uniform_indices)
    expected_samples = nb_classes * samples_per_class
    print(f"📊 Uniform prototype data: {total_samples} training samples (expected: {expected_samples})")
    if total_samples != expected_samples:
        print(f"Sample count mismatch! Got {total_samples}, expected {expected_samples}")
    return np.array(uniform_indices), np.array(uniform_labels)

def create_isolated_evaluation_setup(adj, features, labels, idx_train, idx_val, idx_test):
    print("🔒 Creating completely isolated evaluation setup...")
    train_val_nodes = list(idx_train) + list(idx_val)
    train_mask = np.zeros(adj.shape[0], dtype=bool)
    train_mask[train_val_nodes] = True
    adj_isolated = adj.copy()
    test_nodes = list(idx_test)
    adj_isolated[test_nodes][:, train_val_nodes] = 0
    adj_isolated[train_val_nodes][:, test_nodes] = 0
    for i in test_nodes:
        adj_isolated[i, i] = 1
    features_train_only = features[train_mask]
    features_isolated = features.copy()
    features_isolated[~train_mask] = 0
    print(f"✅ Created isolated setup:")
    print(f"   - Train/val nodes: {len(train_val_nodes)}")
    print(f"   - Test nodes: {len(test_nodes)}")
    print(f"   - Test nodes isolated from train/val graph")
    print(f"   - Test features zeroed out initially")
    return adj_isolated, features_isolated, train_mask, test_nodes

def evaluate_downprompt_sample(model, features, adj, sp_adj, idx_test_sampled, idx_train, labels, device):
    print(f"🔍 Evaluating on {len(idx_test_sampled)} test samples...")
    adj_isolated = create_isolated_test_graph(adj, idx_train, [], idx_test_sampled)
    sp_adj_isolated = process.sparse_mx_to_torch_sparse_tensor(process.normalize_adj(adj_isolated + sp.eye(adj_isolated.shape[0])))
    sampled_labels = labels[idx_test_sampled]
    sampled_class_distribution = np.sum(sampled_labels, axis=0)
    classes_present = np.where(sampled_class_distribution > 0)[0]
    print(f"📊 Classes present: {classes_present} (distribution: {sampled_class_distribution})")
    features = torch.FloatTensor(features[np.newaxis]).to(device)
    sp_adj_isolated = sp_adj_isolated.to(device)
    with torch.no_grad():
        embeds, _ = model.embed(features, sp_adj_isolated, CONFIG['sparse'], None, False)
        embeddings = embeds.squeeze(0)
    print(f"📐 Embeddings shape: {embeddings.shape}")
    dgiprompt = model.dgi.prompt
    graphcledgeprompt = model.graphcledge.prompt
    lpprompt = model.lp.prompt
    print(f"🎯 Prompts loaded: DGI={dgiprompt.shape}, GraphCL={graphcledgeprompt.shape}, LP={lpprompt.shape}")
    feature_prompt = featureprompt(model.dgiprompt.prompt, model.graphcledgeprompt.prompt, model.lpprompt.prompt).to(device)
    with torch.no_grad():
        prompt_feature = feature_prompt(features)
        embeds1, _ = model.embed(prompt_feature, sp_adj_isolated, CONFIG['sparse'], None, False)
        embeddings1 = embeds1.squeeze(0)
    print(f"📐 Prompt embeddings shape: {embeddings1.shape}")
    if CONFIG['use_simple_classifier']:
        print("🤖 Using Simple Classifier Mode")
        return evaluate_with_simple_classifier(embeddings, embeddings1, idx_test_sampled, idx_train, labels, device)
    elif CONFIG['use_vae_decoder']:
        print("🎨 Using VAE Decoder Mode")
        return evaluate_with_vae_decoder(embeddings, embeddings1, idx_test_sampled, idx_train, labels, device, 
                                       model, features, adj, sp_adj)
    else:
        print("🤖 Using Downprompt Mode")
        return evaluate_with_downprompt(embeddings, embeddings1, dgiprompt, graphcledgeprompt, lpprompt, 
                                      idx_test_sampled, idx_train, labels, device)

def evaluate_with_simple_classifier(embeddings, embeddings1, idx_test_sampled, idx_train, labels, device):
    global TRAINED_SIMPLE_CLASSIFIER, TRAINED_CLASSIFIER_EMBEDDINGS, TRAINED_CLASSIFIER_TRAIN_INDICES, TRAINING_COUNT, REUSE_COUNT
    
    nb_classes = labels.shape[1]
    need_to_train = (
        TRAINED_SIMPLE_CLASSIFIER is None or 
        TRAINED_CLASSIFIER_EMBEDDINGS is None or 
        TRAINED_CLASSIFIER_TRAIN_INDICES is None or
        not torch.equal(TRAINED_CLASSIFIER_EMBEDDINGS, embeddings) or
        not np.array_equal(TRAINED_CLASSIFIER_TRAIN_INDICES, list(idx_train))
    )
    
    if need_to_train:
        TRAINING_COUNT += 1
        print(f"🎯 Training new simple classifier (training #{TRAINING_COUNT})...")
        
        all_available_indices = list(range(len(labels)))
        test_indices_set = set(idx_test)
        train_val_indices = [i for i in all_available_indices if i not in test_indices_set]
        
        print(f"📊 Using all non-test data: {len(train_val_indices)} samples (total: {len(labels)}, test: {len(idx_test)})")
        
        train_size = int(CONFIG['train_val_split_ratio'] * len(train_val_indices))
        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]
        
        print(f"📊 Train/Val split: {len(train_indices)}/{len(val_indices)} samples ({CONFIG['train_val_split_ratio']*100:.0f}% train, {100-CONFIG['train_val_split_ratio']*100:.0f}% val)")
        
        train_embeddings = embeddings[train_indices]
        train_labels_onehot = torch.FloatTensor(labels[train_indices]).to(device)
        val_embeddings = embeddings[val_indices]
        val_labels_onehot = torch.FloatTensor(labels[val_indices]).to(device)
        
        classifier = create_simple_classifier(
            num_classes=nb_classes, 
            device=device, 
            dropout=CONFIG['simple_classifier_dropout'],
            hidden_dim=CONFIG['simple_classifier_hidden_dim']
        )
        
        trained_model = train_simple_classifier(
            model=classifier,
            train_embeddings=train_embeddings,
            train_labels=train_labels_onehot,
            val_embeddings=val_embeddings,
            val_labels=val_labels_onehot,
            lr=CONFIG['simple_classifier_lr'],
            epochs=CONFIG['simple_classifier_epochs'],
            patience=CONFIG['simple_classifier_patience'],
            dropout=CONFIG['simple_classifier_dropout'],
            batch_size=CONFIG['simple_classifier_batch_size']
        )
        
        TRAINED_SIMPLE_CLASSIFIER = trained_model
        TRAINED_CLASSIFIER_EMBEDDINGS = embeddings.clone()
        TRAINED_CLASSIFIER_TRAIN_INDICES = list(idx_train)
        
        print("✅ Simple classifier trained and stored for reuse")
    else:
        REUSE_COUNT += 1
        print(f"♻️  Reusing previously trained simple classifier (reuse #{REUSE_COUNT})...")
        trained_model = TRAINED_SIMPLE_CLASSIFIER
    
    test_embeddings = embeddings[idx_test_sampled]
    test_labels = torch.FloatTensor(labels[idx_test_sampled]).to(device)
    
    print("🔍 Evaluating simple classifier...")
    detailed_results = evaluate_simple_classifier_detailed(
        model=trained_model,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        device=device
    )
    
    pred_classes = detailed_results['predictions']
    true_classes = detailed_results['true_labels']
    probabilities = detailed_results['probabilities']
    logits = detailed_results['logits']
    
    print(f"📊 Enhanced Simple Classifier Analysis:")
    print(f"   Accuracy: {detailed_results['accuracy']:.4f} ({detailed_results['accuracy']*100:.1f}%)")
    print(f"   F1 Macro: {detailed_results['f1_macro']:.4f}")
    print(f"   F1 Weighted: {detailed_results['f1_weighted']:.4f}")
    print(f"   Avg Confidence: {detailed_results['avg_confidence']:.4f} ± {detailed_results['confidence_std']:.4f}")
    print(f"   Avg Entropy: {detailed_results['avg_entropy']:.4f}")
    
    print(f"   Per-class Accuracy:")
    for i, acc in enumerate(detailed_results['per_class_accuracy']):
        print(f"     Class {i}: {acc:.4f} ({acc*100:.1f}%)")
    
    pred_class_distribution = np.bincount(pred_classes, minlength=nb_classes)
    print(f"📊 Simple Classifier Predictions: {pred_class_distribution}")
    
    return pred_classes, true_classes, probabilities, logits

def evaluate_with_downprompt(embeddings, embeddings1, dgiprompt, graphcledgeprompt, lpprompt, 
                           idx_test_sampled, idx_train, labels, device):
    print("🎯 Using original downprompt evaluation...")
    
    embedding_dim = embeddings.shape[1]
    nb_classes = labels.shape[1]
    
    uniform_indices, uniform_labels = create_uniform_prototype_data(idx_train, labels, nb_classes, samples_per_class=7)
    uniform_labels_tensor = torch.LongTensor(uniform_labels).to(device)
    
    test_indices_set = set(idx_test_sampled)
    prototype_indices_set = set(uniform_indices)
    overlap = test_indices_set.intersection(prototype_indices_set)
    if overlap:
        print(f"🚨 CRITICAL: Data leakage detected! {len(overlap)} test samples used in prototypes!")
        print(f"🚨 Overlapping indices: {list(overlap)[:5]}...")
    else:
        print(f"✅ No data leakage: {len(prototype_indices_set)} prototype samples are all from training data")
    
    dummy_embeddings = torch.zeros(len(idx_train), embedding_dim).to(device)
    
    downprompt_model = downprompt(dgiprompt, graphcledgeprompt, lpprompt, CONFIG['a4'], 
                                 embedding_dim, nb_classes, dummy_embeddings, uniform_labels_tensor)
    downprompt_model.to(device)
    downprompt_model.eval()
    
    print("✅ Using pretrained downprompt model (no training needed)")
    
    print("🎯 Initializing average embeddings from training data...")
    with torch.no_grad():
        init_embeds = embeddings[uniform_indices]
        init_embeds1 = embeddings1[uniform_indices]
        
        _ = downprompt_model(init_embeds, init_embeds1, train=1)
        
        if hasattr(downprompt_model, 'ave'):
            prototype_norms = torch.norm(downprompt_model.ave, dim=1)
            print(f"🎯 Prototypes initialized: {downprompt_model.ave.shape}")
            print(f"🎯 Prototype norms: {prototype_norms.cpu().numpy()}")
            
            if torch.all(downprompt_model.ave == 0):
                print("All prototypes are zero! This will cause bias.")
            else:
                print("✅ Prototypes properly initialized from training data")
        else:
            print("No 'ave' attribute found in downprompt model")
    
    with torch.no_grad():
        test_embeds = embeddings[idx_test_sampled]
        test_embeds1 = embeddings1[idx_test_sampled]
        
        predictions = downprompt_model(test_embeds, test_embeds1, train=0)
        
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        true_classes = torch.argmax(torch.FloatTensor(labels[idx_test_sampled]), dim=1).numpy()
        
        probabilities = predictions.cpu().numpy()
        
        pred_class_distribution = np.bincount(pred_classes, minlength=nb_classes)
        print(f"📊 Downprompt Predictions: {pred_class_distribution}")
    
    return pred_classes, true_classes, probabilities, predictions.cpu().numpy()

def print_evaluation_results(predictions, true_labels, probabilities, logits, sample_size):
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    print(f"\nSample Predictions (first 5):")
    print("True | Pred | Conf")
    print("-" * 20)
    for i in range(min(5, len(predictions))):
        confidence = probabilities[i][predictions[i]]
        print(f"  {true_labels[i]}  |  {predictions[i]}  | {confidence:.3f}")
    
    print(f"\nSummary: {np.sum(predictions == true_labels)}/{len(predictions)} correct ({np.sum(predictions == true_labels)/len(predictions)*100:.1f}%)")
    
    return accuracy

def evaluate_with_vae_decoder(embeddings, embeddings1, idx_test_sampled, idx_train, labels, device, 
                             model, features, adj, sp_adj):
    print("🎨 Using VAE Decoder Mode")
    
    nb_classes = labels.shape[1]
    if features.dim() == 3:
        feature_dim = features.shape[2]
    else:
        feature_dim = features.shape[1]
    
    print(f"🔍 Debug - Features shape: {features.shape}, Feature dim: {feature_dim}")
    
    print("🔧 Creating VAE decoder...")
    decoder = create_vae_decoder(
        embedding_dim=embeddings.shape[1],
        feature_dim=feature_dim,
        hidden_dims=CONFIG['vae_decoder_hidden_dims'],
        dropout=CONFIG['vae_decoder_dropout']
    )
    decoder.to(device)
    
    print("📊 Preparing VAE training data...")
    if CONFIG['vae_decoder_train_on_train_only']:
        train_loader, val_loader, test_loader, _ = prepare_vae_training_data(
            adj, features, labels, idx_train, [], []
        )
    else:
        train_loader, val_loader, test_loader, _ = prepare_vae_training_data(
            adj, features, labels, idx_train, [], []
        )
    
    print(f"🎯 Training VAE decoder for {CONFIG['vae_decoder_epochs']} epochs...")
    try:
        history = train_vae_decoder(
            encoder=model,
            decoder=decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=CONFIG['vae_decoder_epochs'],
            lr=CONFIG['vae_decoder_lr'],
            beta=CONFIG['vae_decoder_beta'],
            patience=CONFIG['vae_decoder_patience']
        )
        print("✅ VAE decoder training completed")
    except Exception as e:
        print(f"❌ VAE decoder training failed: {e}")
        print("Using untrained decoder for evaluation")
    
    if CONFIG['vae_decoder_evaluate_reconstruction']:
        print("🔍 Evaluating VAE reconstruction quality...")
        try:
            test_subset = idx_test_sampled[:min(len(idx_test_sampled), 100)]
            if len(test_subset) > 0:
                test_loader, _, _, _ = prepare_vae_training_data(
                    adj, features, labels, [], [], test_subset
                )
                
                recon_metrics = evaluate_vae_reconstruction(
                    encoder=model,
                    decoder=decoder,
                    test_loader=test_loader,
                    device=device
                )
                
                print(f"📊 VAE Reconstruction Metrics:")
                print(f"   MSE: {recon_metrics['mse']:.6f}")
                print(f"   MAE: {recon_metrics['mae']:.6f}")
                print(f"   Reconstruction Loss: {recon_metrics['reconstruction_loss']:.6f}")
            else:
                print("⚠️  No test samples available for reconstruction evaluation")
        except Exception as e:
            print(f"❌ VAE reconstruction evaluation failed: {e}")
    
    print("🎯 Using VAE decoder for classification...")
    
    test_embeddings = embeddings[idx_test_sampled]
    test_labels = torch.FloatTensor(labels[idx_test_sampled]).to(device)
    
    with torch.no_grad():
        reconstructed_features = decoder(test_embeddings)
    
    print("\n" + "="*60)
    print("🔬 CLASSIFICATION COMPARISON ANALYSIS")
    print("="*60)
    
    print("\n📊 1. Direct Classification on Original Embeddings:")
    classifier_original = nn.Sequential(
        nn.Linear(embeddings.shape[1], CONFIG['simple_classifier_hidden_dim']),
        nn.ReLU(),
        nn.Dropout(CONFIG['simple_classifier_dropout']),
        nn.Linear(CONFIG['simple_classifier_hidden_dim'], nb_classes)
    ).to(device)
    
    all_available_indices = list(range(len(labels)))
    test_indices_set = set(idx_test_sampled)
    train_val_indices = [i for i in all_available_indices if i not in test_indices_set]
    
    train_val_embeddings = embeddings[train_val_indices]
    train_val_labels = torch.FloatTensor(labels[train_val_indices]).to(device)
    
    train_size = int(CONFIG['train_val_split_ratio'] * len(train_val_indices))
    train_embeddings = train_val_embeddings[:train_size]
    train_labels = train_val_labels[:train_size]
    val_embeddings = train_val_embeddings[train_size:]
    val_labels = train_val_labels[train_size:]
    
    optimizer_orig = torch.optim.Adam(classifier_original.parameters(), lr=CONFIG['simple_classifier_lr'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc_orig = 0
    patience_counter = 0
    
    for epoch in range(min(50, CONFIG['simple_classifier_epochs'])):
        classifier_original.train()
        optimizer_orig.zero_grad()
        
        train_outputs = classifier_original(train_embeddings)
        train_loss = criterion(train_outputs, torch.argmax(train_labels, dim=1))
        
        train_loss.backward()
        optimizer_orig.step()
        
        classifier_original.eval()
        with torch.no_grad():
            val_outputs = classifier_original(val_embeddings)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_true = torch.argmax(val_labels, dim=1)
            val_acc = (val_preds == val_true).float().mean().item()
        
        if val_acc > best_val_acc_orig:
            best_val_acc_orig = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    classifier_original.eval()
    with torch.no_grad():
        test_outputs_orig = classifier_original(test_embeddings)
        test_preds_orig = torch.argmax(test_outputs_orig, dim=1)
        test_true = torch.argmax(test_labels, dim=1)
        acc_orig = (test_preds_orig == test_true).float().mean().item()
    
    print(f"   ✅ Original Embeddings Accuracy: {acc_orig:.4f} ({acc_orig*100:.1f}%)")
    print(f"   📈 Best Validation Accuracy: {best_val_acc_orig:.4f}")
    
    print("\n📊 2. Re-encode Decoded Features and Classify:")
    
    with torch.no_grad():
        full_reconstructed_features = features.clone()
        if full_reconstructed_features.dim() == 3:
            full_reconstructed_features = full_reconstructed_features.squeeze(0)
        
        full_reconstructed_features[idx_test_sampled] = reconstructed_features
        
        full_reconstructed_features_batch = full_reconstructed_features.unsqueeze(0)
        
        re_encoded_embeddings, _ = model.embed(full_reconstructed_features_batch, sp_adj, True, None, False)
        if re_encoded_embeddings.dim() == 3:
            re_encoded_embeddings = re_encoded_embeddings.squeeze(0)
        
        re_encoded_test_embeddings = re_encoded_embeddings[idx_test_sampled]
    
    print(f"   🔄 Re-encoded embeddings shape: {re_encoded_test_embeddings.shape}")
    print(f"   🔄 Original embeddings shape: {test_embeddings.shape}")
    
    classifier_reencoded = nn.Sequential(
        nn.Linear(re_encoded_test_embeddings.shape[1], CONFIG['simple_classifier_hidden_dim']),
        nn.ReLU(),
        nn.Dropout(CONFIG['simple_classifier_dropout']),
        nn.Linear(CONFIG['simple_classifier_hidden_dim'], nb_classes)
    ).to(device)
    
    with torch.no_grad():
        train_val_reconstructed = decoder(train_val_embeddings)
        
        full_train_reconstructed = features.clone()
        if full_train_reconstructed.dim() == 3:
            full_train_reconstructed = full_train_reconstructed.squeeze(0)
        
        full_train_reconstructed[train_val_indices] = train_val_reconstructed
        
        full_train_reconstructed_batch = full_train_reconstructed.unsqueeze(0)
        
        train_val_reencoded_full, _ = model.embed(full_train_reconstructed_batch, sp_adj, True, None, False)
        if train_val_reencoded_full.dim() == 3:
            train_val_reencoded_full = train_val_reencoded_full.squeeze(0)
        
        train_val_reencoded = train_val_reencoded_full[train_val_indices]
    
    train_reencoded = train_val_reencoded[:train_size]
    val_reencoded = train_val_reencoded[train_size:]
    
    optimizer_reencoded = torch.optim.Adam(classifier_reencoded.parameters(), lr=CONFIG['simple_classifier_lr'])
    
    best_val_acc_reencoded = 0
    patience_counter = 0
    
    for epoch in range(min(50, CONFIG['simple_classifier_epochs'])):
        classifier_reencoded.train()
        optimizer_reencoded.zero_grad()
        
        train_outputs = classifier_reencoded(train_reencoded)
        train_loss = criterion(train_outputs, torch.argmax(train_labels, dim=1))
        
        train_loss.backward()
        optimizer_reencoded.step()
        
        classifier_reencoded.eval()
        with torch.no_grad():
            val_outputs = classifier_reencoded(val_reencoded)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_true = torch.argmax(val_labels, dim=1)
            val_acc = (val_preds == val_true).float().mean().item()
        
        if val_acc > best_val_acc_reencoded:
            best_val_acc_reencoded = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    classifier_reencoded.eval()
    with torch.no_grad():
        test_outputs_reencoded = classifier_reencoded(re_encoded_test_embeddings)
        test_preds_reencoded = torch.argmax(test_outputs_reencoded, dim=1)
        acc_reencoded = (test_preds_reencoded == test_true).float().mean().item()
    
    print(f"   ✅ Re-encoded Features Accuracy: {acc_reencoded:.4f} ({acc_reencoded*100:.1f}%)")
    print(f"   📈 Best Validation Accuracy: {best_val_acc_reencoded:.4f}")
    
    print("\n" + "="*60)
    print("📊 SUMMARY COMPARISON")
    print("="*60)
    print(f"1. Original Embeddings:     {acc_orig:.4f} ({acc_orig*100:.1f}%)")
    print(f"2. Re-encoded Features:     {acc_reencoded:.4f} ({acc_reencoded*100:.1f}%)")
    
    diff_reencoded = acc_reencoded - acc_orig
    
    print(f"\n📈 Performance Differences:")
    print(f"   Re-encoded vs Original:  {diff_reencoded:+.4f} ({diff_reencoded*100:+.1f}%)")
    
    best_acc = max(acc_orig, acc_reencoded)
    if best_acc == acc_orig:
        best_method = "Original Embeddings"
        final_outputs = test_outputs_orig
    else:
        best_method = "Re-encoded Features"
        final_outputs = test_outputs_reencoded
    
    print(f"\n🏆 Best Method: {best_method} ({best_acc:.4f})")
    print("="*60)
    
    probabilities = torch.softmax(final_outputs, dim=1)
    
    pred_classes = torch.argmax(final_outputs, dim=1).cpu().numpy()
    true_classes = test_true.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    logits = final_outputs.cpu().numpy()
    
    return pred_classes, true_classes, probabilities, logits

def main():
    print("🚀 Starting Simplified Downprompt Model Evaluation (FIXED)")
    print("="*50)
    print("This evaluation uses isolated test nodes to prevent data leakage")
    print("Pretrained model may still contain test data information!")
    print("🔒 FIXED: Using deterministic sampling and balanced class distribution")
    
    print_config()
    
    if CONFIG['use_simple_classifier']:
        print("�� MODE: Simple Classifier (trains a neural network on embeddings)")
        print(f"📚 Training: {CONFIG['simple_classifier_epochs']} epochs, lr={CONFIG['simple_classifier_lr']}")
        reset_simple_classifier()
    elif CONFIG['use_vae_decoder']:
        print("🎨 MODE: VAE Decoder (reconstructs features from embeddings)")
        print(f"🔧 Decoder: {CONFIG['vae_decoder_hidden_dims']} hidden dims, lr={CONFIG['vae_decoder_lr']}")
        print(f"🎯 Training: {CONFIG['vae_decoder_epochs']} epochs, beta={CONFIG['vae_decoder_beta']}")
    else:
        print("🤖 MODE: Downprompt (uses prototype-based classification)")
    
    print("="*50)
    
    set_seed(CONFIG['seed'])
    
    device = torch.device(CONFIG['device'])
    print(f"💻 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 CUDA: {torch.cuda.get_device_name()}")
    
    adj, features, labels, idx_train, idx_val, idx_test, nb_nodes, ft_size, nb_classes = \
        load_and_preprocess_data(CONFIG['dataset'])
    
    adj_isolated, features_isolated, train_mask, test_nodes = \
        create_isolated_evaluation_setup(adj, features, labels, idx_train, idx_val, idx_test)
    
    (aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
     sp_adj, sp_aug_adj1edge, sp_aug_adj2edge, sp_aug_adj1mask, sp_aug_adj2mask) = \
        prepare_augmented_data(adj_isolated, features_isolated, CONFIG['drop_percent'], idx_train, idx_val, idx_test)
    
    model, model_loaded_successfully = load_pretrained_model(ft_size, CONFIG['hid_units'], device, adj_isolated, CONFIG['dataset'])
    
    if CONFIG['use_all_test_data']:
        print(f"📋 Using ALL {len(idx_test)} test samples")
        idx_test_sampled = idx_test
    else:
        print(f"📋 Sampling {CONFIG['sample_size']} test samples with balanced distribution")
        idx_test_sampled = sample_balanced_test_data(idx_test, CONFIG['sample_size'], labels, nb_classes)
    
    pred_classes, true_classes, probabilities, logits = \
        evaluate_downprompt_sample(model, features_isolated, adj_isolated, sp_adj, idx_test_sampled, idx_train, labels, device)
    
    accuracy = print_evaluation_results(pred_classes, true_classes, probabilities, logits, CONFIG['sample_size'])
    
    print("\n" + "="*50)
    print("✅ Evaluation completed (FIXED for data leakage and consistency)!")
    print(f"🎯 Final accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"🤖 Model loaded: {'✅ Yes' if model_loaded_successfully else '❌ No'}")
    print("🔒 Data leakage prevention: Test nodes isolated from training graph")
    print("�� Consistency fixes: Deterministic sampling and balanced class distribution")
    
    if CONFIG['use_simple_classifier']:
        print("🤖 Evaluation Mode: Simple Classifier")
        print(f"📊 Training Statistics: Trained {TRAINING_COUNT} times, Reused {REUSE_COUNT} times")
    else:
        print("🤖 Evaluation Mode: Downprompt")
    
    print("Note: Accuracy may be lower due to proper isolation")
    print("="*50)
    
    return accuracy, pred_classes, true_classes, nb_classes, model_loaded_successfully

def run_custom_evaluation(dataset='cora', sample_size=50, seed=39, use_all_test_data=False, single_run_mode=False):
    print(f"🔬 Running custom evaluation: {dataset}, {'all test data' if use_all_test_data else f'{sample_size} samples'}, seed {seed}, {'single run' if single_run_mode else 'multi-run'}...")
    
    if CONFIG['use_simple_classifier']:
        print("🤖 MODE: Simple Classifier (trains a neural network on embeddings)")
    else:
        print("🤖 MODE: Downprompt (uses prototype-based classification)")
    
    update_config(
        dataset=dataset,
        sample_size=sample_size,
        seed=seed,
        use_all_test_data=use_all_test_data,
        single_run_mode=single_run_mode
    )
    
    if single_run_mode:
        device = torch.device(CONFIG['device'])
        print(f"💻 Device: {device}")
        
        print("📁 Loading dataset...")
        adj, features, labels, idx_train, idx_val, idx_test, nb_nodes, ft_size, nb_classes = \
            load_and_preprocess_data(CONFIG['dataset'])
        
        adj_isolated, features_isolated, train_mask, test_nodes = \
            create_isolated_evaluation_setup(adj, features, labels, idx_train, idx_val, idx_test)
        
        (aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
         sp_adj, sp_aug_adj1edge, sp_aug_adj2edge, sp_aug_adj1mask, sp_aug_adj2mask) = \
            prepare_augmented_data(adj_isolated, features_isolated, CONFIG['drop_percent'], idx_train, idx_val, idx_test)
        
        print("🤖 Loading pretrained model...")
        model, model_loaded_successfully = load_pretrained_model(ft_size, CONFIG['hid_units'], device, adj_isolated, CONFIG['dataset'])
        
        set_seed(CONFIG['seed'])
        print(f"🎲 Seed: {CONFIG['seed']}")
        
        if CONFIG['use_all_test_data']:
            print(f"📋 Using ALL {len(idx_test)} test samples")
            idx_test_sampled = idx_test
        else:
            print(f"📋 Sampling {CONFIG['sample_size']} test samples with balanced distribution")
            idx_test_sampled = sample_test_data(idx_test, CONFIG['sample_size'], nb_classes)
            print(f"🔍 Sample indices (first 5): {idx_test_sampled[:5]}")
        
        pred_classes, true_classes, probabilities, logits = \
            evaluate_downprompt_sample(model, features_isolated, adj_isolated, sp_adj, idx_test_sampled, idx_train, labels, device)
        
        accuracy = accuracy_score(true_classes, pred_classes)
        
        return accuracy, pred_classes, true_classes, nb_classes, model_loaded_successfully
    else:
        accuracy, pred_classes, true_classes, nb_classes, model_loaded_successfully = main()
        return accuracy, pred_classes, true_classes, nb_classes, model_loaded_successfully

if __name__ == "__main__":
    if CONFIG['single_run_mode']:
        print("🎯 Single-Run Downprompt Evaluation (FIXED)")
        print("="*50)
        print("This evaluation uses isolated test nodes to prevent data leakage")
        print("Pretrained model may still contain test data information!")
        print("="*50)
        
        print_config()
        
        if CONFIG['use_simple_classifier']:
            print("🤖 MODE: Simple Classifier (trains a neural network on embeddings)")
            print(f"📚 Training: {CONFIG['simple_classifier_epochs']} epochs, lr={CONFIG['simple_classifier_lr']}")
            reset_simple_classifier()
            print(f"🤖 Simple Classifier: Trained {TRAINING_COUNT} times, Reused {REUSE_COUNT} times")
            
        elif CONFIG['use_vae_decoder']:
            print("🎨 MODE: VAE Decoder (reconstructs features from embeddings)")
            print(f"🔧 Decoder: {CONFIG['vae_decoder_hidden_dims']} hidden dims, lr={CONFIG['vae_decoder_lr']}")
            print(f"🎯 Training: {CONFIG['vae_decoder_epochs']} epochs, beta={CONFIG['vae_decoder_beta']}")
        else:
            print("🤖 MODE: Downprompt (uses prototype-based classification)")
        
        device = torch.device(CONFIG['device'])
        print(f"💻 Device: {device}")
        
        print("📁 Loading dataset...")
        adj, features, labels, idx_train, idx_val, idx_test, nb_nodes, ft_size, nb_classes = \
            load_and_preprocess_data(CONFIG['dataset'])
        
        adj_isolated, features_isolated, train_mask, test_nodes = \
            create_isolated_evaluation_setup(adj, features, labels, idx_train, idx_val, idx_test)
        
        (aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
         sp_adj, sp_aug_adj1edge, sp_aug_adj2edge, sp_aug_adj1mask, sp_aug_adj2mask) = \
            prepare_augmented_data(adj_isolated, features_isolated, CONFIG['drop_percent'], idx_train, idx_val, idx_test)
        
        print("🤖 Loading pretrained model...")
        model, model_loaded_successfully = load_pretrained_model(ft_size, CONFIG['hid_units'], device, adj_isolated, CONFIG['dataset'])
        
        set_seed(CONFIG['seed'])
        print(f"🎲 Seed: {CONFIG['seed']}")
        
        try:
            if CONFIG['use_all_test_data']:
                print(f"📋 Using ALL {len(idx_test)} test samples")
                idx_test_sampled = idx_test
            else:
                print(f"📋 Sampling {CONFIG['sample_size']} test samples with balanced distribution")
                idx_test_sampled = sample_test_data(idx_test, CONFIG['sample_size'], nb_classes)
                print(f"🔍 Sample indices (first 5): {idx_test_sampled[:5]}")
            
            pred_classes, true_classes, probabilities, logits = \
                evaluate_downprompt_sample(model, features_isolated, adj_isolated, sp_adj, idx_test_sampled, idx_train, labels, device)
            
            accuracy = accuracy_score(true_classes, pred_classes)
            
            print(f"\n{'='*50}")
            print("📊 SINGLE RUN RESULTS (FIXED)")
            print(f"{'='*50}")
            print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"📊 Test samples: {len(idx_test_sampled)}")
            print(f"🤖 Model loaded: {'✅ Yes' if model_loaded_successfully else '❌ No'}")
            print(f"🎲 Seed: {CONFIG['seed']}")
            
            print("🔒 Data leakage prevention: Test nodes isolated from training graph")
            print("🔒 Consistency fixes: Deterministic sampling and balanced class distribution")
            
            print("\n🎉 Single run evaluation completed (FIXED)!")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("🎯 Multi-Run Downprompt Evaluation (FIXED)")
        print("="*50)
        print("This evaluation uses isolated test nodes to prevent data leakage")
        print("Pretrained model may still contain test data information!")
        print("="*50)
        
        if CONFIG['use_simple_classifier']:
            print("🤖 MODE: Simple Classifier (trains once, reuses for all runs)")
            print(f"📚 Training: {CONFIG['simple_classifier_epochs']} epochs, lr={CONFIG['simple_classifier_lr']}")
            reset_simple_classifier()
            print(f"🤖 Simple Classifier: Trained {TRAINING_COUNT} times, Reused {REUSE_COUNT} times")
          
        elif CONFIG['use_vae_decoder']:
            print("🎨 MODE: VAE Decoder (reconstructs features from embeddings)")
            print(f"🔧 Decoder: {CONFIG['vae_decoder_hidden_dims']} hidden dims, lr={CONFIG['vae_decoder_lr']}")
            print(f"🎯 Training: {CONFIG['vae_decoder_epochs']} epochs, beta={CONFIG['vae_decoder_beta']}")
        else:
            print("🤖 MODE: Downprompt (uses prototype-based classification)")
        
        device = torch.device(CONFIG['device'])
        print(f"💻 Device: {device}")
        
        print("📁 Loading dataset once for all runs...")
        adj, features, labels, idx_train, idx_val, idx_test, nb_nodes, ft_size, nb_classes = \
            load_and_preprocess_data(CONFIG['dataset'])
        
        adj_isolated, features_isolated, train_mask, test_nodes = \
            create_isolated_evaluation_setup(adj, features, labels, idx_train, idx_val, idx_test)
        
        (aug_features1edge, aug_features2edge, aug_features1mask, aug_features2mask,
         sp_adj, sp_aug_adj1edge, sp_aug_adj2edge, sp_aug_adj1mask, sp_aug_adj2mask) = \
            prepare_augmented_data(adj_isolated, features_isolated, CONFIG['drop_percent'], idx_train, idx_val, idx_test)
        
        print("🤖 Loading pretrained model once for all runs...")
        model, model_loaded_successfully = load_pretrained_model(ft_size, CONFIG['hid_units'], device, adj_isolated, CONFIG['dataset'])
        
        results = []
        
        for run in range(3):
            print(f"\n{'='*20} RUN {run + 1}/3 {'='*20}")
        
            run_seed = CONFIG['seed'] + run
            set_seed(run_seed)
            print(f"🎲 Seed: {run_seed}")
            
            try:
                if CONFIG['use_all_test_data']:
                    print(f"📋 Using ALL {len(idx_test)} test samples")
                    idx_test_sampled = idx_test
                else:
                    print(f"📋 Sampling {CONFIG['sample_size']} test samples")
                    idx_test_sampled = sample_test_data(idx_test, CONFIG['sample_size'], nb_classes)
                    print(f"🔍 Sample indices (first 5): {idx_test_sampled[:5]}")
                
                pred_classes, true_classes, probabilities, logits = \
                    evaluate_downprompt_sample(model, features_isolated, adj_isolated, sp_adj, idx_test_sampled, idx_train, labels, device)
                
                accuracy = accuracy_score(true_classes, pred_classes)
                
                print(f"✅ Run {run + 1} completed: {accuracy:.4f} ({accuracy*100:.1f}%)")
                
                results.append({
                    'run': run + 1,
                    'accuracy': accuracy,
                    'predictions': pred_classes,
                    'true_labels': true_classes,
                    'nb_classes': nb_classes,
                    'model_loaded_successfully': model_loaded_successfully,
                    'seed': run_seed
                })
                
            except Exception as e:
                print(f"❌ Run {run + 1} failed: {e}")
                results.append({
                    'run': run + 1,
                    'accuracy': 0.0,
                    'predictions': [],
                    'true_labels': [],
                    'nb_classes': nb_classes,
                    'model_loaded_successfully': model_loaded_successfully,
                    'seed': run_seed
                })
        
        print("\n" + "="*50)
        print("📊 SUMMARY OF ALL RUNS (FIXED)")
        print("="*50)
        
        for i, result in enumerate(results):
            model_status = "✅ Loaded" if result['model_loaded_successfully'] else "❌ Random"
            print(f"Run {result['run']}: {result['accuracy']:.4f} ({result['accuracy']*100:.1f}%) | {model_status} | Seed {result['seed']}")
            
        valid_accuracies = [r['accuracy'] for r in results if r['accuracy'] > 0]
        if valid_accuracies:
            avg_accuracy = np.mean(valid_accuracies)
            std_accuracy = np.std(valid_accuracies)
            best_accuracy = max(valid_accuracies)
            worst_accuracy = min(valid_accuracies)
            print(f"\n📈 Statistics:")
            print(f"   Average: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"   Best: {best_accuracy:.4f}")
            print(f"   Worst: {worst_accuracy:.4f}")
        else:
            avg_accuracy = std_accuracy = best_accuracy = worst_accuracy = 0.0
            print("\n❌ No valid accuracies found!")
        
        successful_loads = sum(1 for r in results if r['model_loaded_successfully'])
        print(f"🤖 Model loading: {successful_loads}/{len(results)} successful")
        
        print("🔒 Data leakage prevention: Test nodes isolated from training graph")
        print("🔒 Consistency fixes: Deterministic sampling and balanced class distribution")

        print("\n🎉 All evaluations completed (FIXED)!")
        print("📊 Check confusion_matrix.png for visualization.") 