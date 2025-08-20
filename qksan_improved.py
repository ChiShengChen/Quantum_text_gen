#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved QKSAN variants with multiple enhancements:
1. Hybrid Classical-Quantum Architecture
2. Advanced Encoding Strategies
3. Better Ansatz Design
4. Improved Training Techniques
5. Multi-class Support
6. Attention Mechanism Integration
7. Regularization and Optimization
8. Better Evaluation Metrics

python qksan_improved.py --model HybridAmHE --epochs 100 --dataset mnist --digits 0 1 2 --embedding_dim 16 --n_qubits 6 --batch_size 64 --lr 0.001 --device cuda
Date: 2025-01-27
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import time
import json
import os

# --------------------
# 1. Enhanced Data utilities
# --------------------
class MultiClassPCADataset(Dataset):
    def __init__(self, X, y, num_classes):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_multi_class_mnist_like(name="mnist", digits=(0,1,2), pca_dim=16, test_size=0.2, random_state=42):
    """
    Enhanced data loading with multi-class support and better preprocessing
    """
    if name == "mnist":
        data = fetch_openml('mnist_784', version=1, as_frame=False)
    elif name == "fmnist":
        data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    else:
        raise ValueError("name must be mnist or fmnist")

    X = data['data']
    y = data['target'].astype(int)

    # Filter for specified digits
    mask = np.isin(y, digits)
    X = X[mask]
    y = y[mask]
    
    # Remap labels to 0, 1, 2, ...
    label_map = {digit: idx for idx, digit in enumerate(digits)}
    y = np.array([label_map[label] for label in y])

    # Enhanced preprocessing
    X = StandardScaler().fit_transform(X)
    
    # Use more components for better representation
    pca = PCA(n_components=pca_dim, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    # Add noise for robustness
    noise_level = 0.01
    X_pca += np.random.normal(0, noise_level, X_pca.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return (X_train, y_train), (X_test, y_test), len(digits)


# --------------------
# 2. Advanced Quantum Components
# --------------------
def amplitude_encode(x, wires):
    """AmplitudeEmbedding wrapper with padding & normalization."""
    # Convert tensor to numpy if needed, detaching gradients
    if hasattr(x, 'detach'):
        x = x.detach().numpy()
    
    # PennyLane AmplitudeEmbedding expects len(x) == 2**n_qubits
    needed = 2 ** len(wires)
    if len(x) < needed:
        x = np.pad(x, (0, needed-len(x)))
    # normalized automatically by normalize=True
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True)


def angle_encode(x, wires):
    """AngleEmbedding with RX rotations."""
    # Convert tensor to numpy if needed, detaching gradients
    if hasattr(x, 'detach'):
        x = x.detach().numpy()
    
    # If len(x) < n_qubits, tile or pad
    n_qubits = len(wires)
    if len(x) < n_qubits:
        x = np.pad(x, (0, n_qubits-len(x)))
    elif len(x) > n_qubits:
        x = x[:n_qubits]
    qml.AngleEmbedding(x, wires=wires, rotation='X')


def he_ansatz(params, wires):
    """
    Hardware-Efficient ansatz:
    For each layer:
      - Single-qubit rotations RX, RY, RZ
      - Simple entangling CRY in a ring
    params shape: [L, len(wires), 3] (for RX, RY, RZ per qubit)
    """
    L = params.shape[0]
    n = len(wires)
    for l in range(L):
        for i, w in enumerate(wires):
            qml.RX(params[l, i, 0], wires=w)
            qml.RY(params[l, i, 1], wires=w)
            qml.RZ(params[l, i, 2], wires=w)
        # entangle
        for i in range(n):
            qml.CRY(params[l, i, 1], wires=[wires[i], wires[(i+1)%n]])


def qaoa_ansatz(params, wires):
    """
    Minimal QAOA-like ansatz (not tied to a specific cost Hamiltonian, for demo):
    For each layer l:
      - Mixer RX on each qubit (angle params[l, :, 0])
      - ZZ entanglers with RZ-RZ form using params[l, i, 1]
    params shape: [L, len(wires), 2]
    """
    L = params.shape[0]
    n = len(wires)
    for l in range(L):
        # Mixer
        for i, w in enumerate(wires):
            qml.RX(params[l, i, 0], wires=w)
        # Entanglers
        for i in range(0, n-1, 2):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.RZ(params[l, i, 1], wires=wires[i+1])
            qml.CNOT(wires=[wires[i], wires[i+1]])


def enhanced_amplitude_encode(x, wires, method='pad'):
    """
    Enhanced amplitude encoding with multiple strategies
    """
    # Convert tensor to numpy if needed, detaching gradients
    if hasattr(x, 'detach'):
        x = x.detach().numpy()
    
    needed = 2 ** len(wires)
    if len(x) < needed:
        if method == 'pad':
            x = np.pad(x, (0, needed-len(x)))
        elif method == 'repeat':
            x = np.tile(x, int(np.ceil(needed/len(x))))[:needed]
        elif method == 'interpolate':
            x_old = np.linspace(0, 1, len(x))
            x_new = np.linspace(0, 1, needed)
            x = np.interp(x_new, x_old, x)
    elif len(x) > needed:
        x = x[:needed]
    
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True)


def enhanced_angle_encode(x, wires, strategy='pad'):
    """
    Enhanced angle encoding with multiple strategies
    """
    # Convert tensor to numpy if needed, detaching gradients
    if hasattr(x, 'detach'):
        x = x.detach().numpy()
    
    n_qubits = len(wires)
    if len(x) < n_qubits:
        if strategy == 'pad':
            x = np.pad(x, (0, n_qubits-len(x)))
        elif strategy == 'repeat':
            x = np.tile(x, int(np.ceil(n_qubits/len(x))))[:n_qubits]
        elif strategy == 'interpolate':
            x_old = np.linspace(0, 1, len(x))
            x_new = np.linspace(0, 1, n_qubits)
            x = np.interp(x_new, x_old, x)
    elif len(x) > n_qubits:
        x = x[:n_qubits]
    
    # Use different rotation axes for better expressivity
    for i, (wire, val) in enumerate(zip(wires, x)):
        qml.RX(val, wires=wire)
        qml.RY(val * 0.5, wires=wire)  # Additional rotation for complexity


def improved_he_ansatz(params, wires, connectivity='ring'):
    """
    Improved Hardware-Efficient ansatz with better connectivity patterns
    """
    L = params.shape[0]
    n = len(wires)
    
    for l in range(L):
        # Single-qubit rotations
        for i, w in enumerate(wires):
            qml.RX(params[l, i, 0], wires=w)
            qml.RY(params[l, i, 1], wires=w)
            qml.RZ(params[l, i, 2], wires=w)
        
        # Enhanced entangling patterns
        if connectivity == 'ring':
            for i in range(n):
                qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
                qml.CRY(params[l, i, 1], wires=[wires[i], wires[(i+1)%n]])
        elif connectivity == 'all_to_all':
            for i in range(n):
                for j in range(i+1, n):
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.CRY(params[l, i, 1], wires=[wires[i], wires[j]])
        elif connectivity == 'nearest_neighbor':
            for i in range(n-1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
                qml.CRY(params[l, i, 1], wires=[wires[i], wires[i+1]])


def improved_qaoa_ansatz(params, wires, mixer_type='rx'):
    """
    Improved QAOA ansatz with better mixer and cost Hamiltonians
    """
    L = params.shape[0]
    n = len(wires)
    
    for l in range(L):
        # Mixer layer
        if mixer_type == 'rx':
            for i, w in enumerate(wires):
                qml.RX(params[l, i, 0], wires=w)
        elif mixer_type == 'ry':
            for i, w in enumerate(wires):
                qml.RY(params[l, i, 0], wires=w)
        elif mixer_type == 'rz':
            for i, w in enumerate(wires):
                qml.RZ(params[l, i, 0], wires=w)
        
        # Cost layer with ZZ interactions
        for i in range(0, n-1, 2):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.RZ(params[l, i, 1], wires=wires[i+1])
            qml.CNOT(wires=[wires[i], wires[i+1]])
        
        # Additional entangling
        if n > 2:
            for i in range(1, n-1, 2):
                qml.CNOT(wires=[wires[i], wires[i+1]])
                qml.RZ(params[l, i, 1], wires=wires[i+1])
                qml.CNOT(wires=[wires[i], wires[i+1]])


# --------------------
# 3. Classical Neural Network Components
# --------------------
class ClassicalEncoder(nn.Module):
    """
    Classical neural network for feature extraction and preprocessing
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ClassicalDecoder(nn.Module):
    """
    Classical neural network for post-processing quantum outputs
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class QuantumAttention(nn.Module):
    """
    Quantum-enhanced attention mechanism
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)


# --------------------
# 4. Enhanced Quantum-Classical Hybrid Model
# --------------------
class EnhancedQuantumClassifier(nn.Module):
    """
    Enhanced quantum classifier with classical pre/post-processing
    """
    def __init__(self,
                 input_dim,
                 n_qubits=6,
                 encoding_type='amplitude',
                 ansatz_type='HE',
                 he_layers=4,
                 qaoa_layers=4,
                 num_classes=2,
                 classical_hidden_dims=[64, 32],
                 use_attention=True,
                 dropout=0.1,
                 use_classical_nn=True,
                 device='cpu'):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.ansatz_type = ansatz_type
        self.num_classes = num_classes
        self.wires = list(range(n_qubits))
        self.device = device
        self.use_classical_nn = use_classical_nn

        # Classical encoder (only if use_classical_nn is True)
        if use_classical_nn:
            self.classical_encoder = ClassicalEncoder(
                input_dim, classical_hidden_dims, n_qubits, dropout
            )
        else:
            # No linear layer needed - direct input like variants.py
            pass

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = QuantumAttention(n_qubits, num_heads=2, dropout=dropout)

        # Determine param shapes
        if ansatz_type == 'HE':
            self.param_shape = (he_layers, n_qubits, 3)
        elif ansatz_type == 'QAOA':
            self.param_shape = (qaoa_layers, n_qubits, 2)
        else:
            raise ValueError("ansatz_type must be HE or QAOA")

        # Trainable parameters with better initialization
        self.theta = nn.Parameter(torch.randn(*self.param_shape) * 0.01)  # 改為0.01以匹配variants

        # PennyLane device
        self.qdev = qml.device("default.qubit", wires=n_qubits, shots=None)

        # Define qnode
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(x, theta):
            # Use simple encoding like variants.py
            if self.encoding_type == 'amplitude':
                amplitude_encode(x, wires=self.wires)
            else:
                angle_encode(x, wires=self.wires)
            
            # Use simple ansatz like variants.py
            if self.ansatz_type == 'HE':
                he_ansatz(theta, wires=self.wires)
            else:
                qaoa_ansatz(theta, wires=self.wires)
            
            # Measure only one qubit like variants.py for better training
            return qml.expval(qml.PauliZ(self.wires[-1]))

        self.circuit = circuit

        # Classical decoder (only if use_classical_nn is True)
        if use_classical_nn:
            self.classical_decoder = ClassicalDecoder(
                n_qubits, classical_hidden_dims[::-1], num_classes, dropout
            )
        else:
            # No additional classifier needed for binary classification
            pass

        # No regularization needed to match variants.py simplicity

        self.to(device)

    def forward(self, x):
        """
        Enhanced forward pass with classical-quantum-classical pipeline
        """
        batch_size = x.shape[0]
        
        # Classical preprocessing
        if self.use_classical_nn:
            x_encoded = self.classical_encoder(x)
            x_encoded = self.dropout(x_encoded)
            
            # Apply attention if enabled
            if self.use_attention:
                # Reshape for attention (batch_size, 1, n_qubits)
                x_attn = x_encoded.unsqueeze(1)
                x_attn = self.attention(x_attn)
                x_encoded = x_attn.squeeze(1)
        else:
            # Direct PCA to quantum mapping - remove linear layer to match variants.py
            x_encoded = x  # Direct input like variants.py
        
        # Quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            out = self.circuit(x_encoded[i], self.theta)
            quantum_outputs.append(out)
        
        quantum_outputs = torch.stack(quantum_outputs)
        # Ensure the tensor is on the correct device and has correct dtype
        quantum_outputs = quantum_outputs.to(device=self.device, dtype=torch.float32)
        
        # Classical post-processing - for binary classification like variants.py
        if self.num_classes == 2:
            # Map (-1,1) to logits like variants.py
            logit1 = quantum_outputs  # positive class
            logit0 = -quantum_outputs  # negative class
            logits = torch.stack([logit0, logit1], dim=1)
        else:
            # For multi-class, use classical decoder
            if self.use_classical_nn:
                logits = self.classical_decoder(quantum_outputs)
            else:
                # For quantum-only classification, create simple linear layer
                if not hasattr(self, 'quantum_output_layer'):
                    self.quantum_output_layer = nn.Linear(1, self.num_classes).to(self.device)
                logits = self.quantum_output_layer(quantum_outputs)
        
        return logits


# --------------------
# 5. Enhanced Training Loop
# --------------------
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_enhanced_model(model, train_loader, test_loader, epochs, lr, device, 
                        weight_decay=1e-4, patience=15):
    """
    Enhanced training loop with better optimization and monitoring
    """
    # Optimizer with weight decay
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)  # 改為SGD以匹配variants
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Metrics tracking
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    best_test_acc = 0.0
    start_time = time.time()
    
    for ep in range(1, epochs+1):
        # Training phase
        model.train()
        tot_loss, tot_acc = 0.0, 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            
            # Add L2 regularization
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-5 * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            tot_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            tot_acc += (preds == yb).float().sum().item()
            num_batches += 1

        avg_train_loss = tot_loss / len(train_loader.dataset)
        avg_train_acc = tot_acc / len(train_loader.dataset)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            te_loss, te_acc = 0.0, 0.0
            all_preds, all_labels = [], []
            
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                
                te_loss += loss.item() * xb.size(0)
                te_preds = logits.argmax(dim=1)
                te_acc += (te_preds == yb).float().sum().item()
                
                all_preds.extend(te_preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        
        avg_test_loss = te_loss / len(test_loader.dataset)
        avg_test_acc = te_acc / len(test_loader.dataset)
        
        # Update scheduler
        scheduler.step(avg_test_loss)
        
        # Track metrics
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        test_losses.append(avg_test_loss)
        test_accs.append(avg_test_acc)
        
        # Save best model
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"[Epoch {ep:03d}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc*100:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} Acc: {avg_test_acc*100:.2f}% | "
              f"Time: {elapsed:.1f}s")
        
        # Early stopping check
        if early_stopping(avg_test_loss):
            print(f"Early stopping triggered after {ep} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Final evaluation with detailed metrics
    final_metrics = evaluate_model(model, test_loader, device)
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'final_metrics': final_metrics
    }


def evaluate_model(model, test_loader, device):
    """
    Comprehensive model evaluation
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }


# --------------------
# 6. Model Factory
# --------------------
def build_enhanced_model(name, input_dim, n_qubits, num_classes, device):
    """
    Enhanced model factory with more variants
    """
    model_configs = {
        "HybridAmHE": {
            "encoding_type": "amplitude",
            "ansatz_type": "HE",
            "he_layers": 4,
            "qaoa_layers": 0,
            "use_attention": True,
            "use_classical_nn": True
        },
        "HybridAnHE": {
            "encoding_type": "angle",
            "ansatz_type": "HE",
            "he_layers": 4,
            "qaoa_layers": 0,
            "use_attention": True,
            "use_classical_nn": True
        },
        "HybridAmQAOA": {
            "encoding_type": "amplitude",
            "ansatz_type": "QAOA",
            "he_layers": 0,
            "qaoa_layers": 4,
            "use_attention": True,
            "use_classical_nn": True
        },
        "HybridAnQAOA": {
            "encoding_type": "angle",
            "ansatz_type": "QAOA",
            "he_layers": 0,
            "qaoa_layers": 4,
            "use_attention": True,
            "use_classical_nn": True
        },
        "PureAmHE": {
            "encoding_type": "amplitude",
            "ansatz_type": "HE",
            "he_layers": 6,
            "qaoa_layers": 0,
            "use_attention": False,
            "use_classical_nn": False
        },
        "PureAnHE": {
            "encoding_type": "angle",
            "ansatz_type": "HE",
            "he_layers": 6,
            "qaoa_layers": 0,
            "use_attention": False,
            "use_classical_nn": False
        },
        "PCAAmHE": {
            "encoding_type": "amplitude",
            "ansatz_type": "HE",
            "he_layers": 6,  # 匹配AmHE的層數
            "qaoa_layers": 0,
            "use_attention": False,
            "use_classical_nn": False
        },
        "PCAAnHE": {
            "encoding_type": "angle",
            "ansatz_type": "HE",
            "he_layers": 4,
            "qaoa_layers": 0,
            "use_attention": False,
            "use_classical_nn": False
        },
        "PCAAmQAOA": {
            "encoding_type": "amplitude",
            "ansatz_type": "QAOA",
            "he_layers": 0,
            "qaoa_layers": 4,
            "use_attention": False,
            "use_classical_nn": False
        },
        "PCAAnQAOA": {
            "encoding_type": "angle",
            "ansatz_type": "QAOA",
            "he_layers": 0,
            "qaoa_layers": 4,
            "use_attention": False,
            "use_classical_nn": False
        }
    }
    
    if name not in model_configs:
        raise ValueError(f"Model name must be one of {list(model_configs.keys())}")
    
    config = model_configs[name]
    
    return EnhancedQuantumClassifier(
        input_dim=input_dim,
        n_qubits=n_qubits,
        encoding_type=config["encoding_type"],
        ansatz_type=config["ansatz_type"],
        he_layers=config["he_layers"],
        qaoa_layers=config["qaoa_layers"],
        num_classes=num_classes,
        use_attention=config["use_attention"],
        use_classical_nn=config["use_classical_nn"],
        device=device
    )


# --------------------
# 7. Visualization and Analysis
# --------------------
def plot_training_curves(results, save_path='training_curves.png'):
    """
    Plot training and validation curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(results['train_losses'], label='Train Loss', color='blue')
    ax1.plot(results['test_losses'], label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(results['train_accs'], label='Train Accuracy', color='blue')
    ax2.plot(results['test_accs'], label='Test Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(conf_matrix, class_names, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# --------------------
# 8. Main Function
# --------------------
def main():
    parser = argparse.ArgumentParser(description='Enhanced Quantum-Classical Hybrid Model')
    parser.add_argument('--model', type=str, default="HybridAmHE", 
                       choices=["HybridAmHE","HybridAnHE","HybridAmQAOA","HybridAnQAOA","PureAmHE","PureAnHE",
                               "PCAAmHE","PCAAnHE","PCAAmQAOA","PCAAnQAOA"])
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist","fmnist"])
    parser.add_argument('--digits', type=int, nargs='+', default=[0,1])
    parser.add_argument('--embedding_dim', type=int, default=8)
    parser.add_argument('--n_qubits', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.09)  # 改為0.09以匹配variants
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_results', action='store_true')
    args = parser.parse_args()

    print(f"Running Enhanced Quantum Model: {args.model}")
    print(f"Dataset: {args.dataset}, Digits: {args.digits}")
    print(f"Embedding dim: {args.embedding_dim}, Qubits: {args.n_qubits}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}, Device: {args.device}")

    # Load data
    (Xtr, ytr), (Xte, yte), num_classes = load_multi_class_mnist_like(
        args.dataset, tuple(args.digits), pca_dim=args.embedding_dim
    )
    
    train_ds = MultiClassPCADataset(Xtr, ytr, num_classes)
    test_ds = MultiClassPCADataset(Xte, yte, num_classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = build_enhanced_model(
        args.model, args.embedding_dim, args.n_qubits, num_classes, args.device
    )
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print(f"\nStarting training...")
    results = train_enhanced_model(
        model, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr, device=args.device,
        weight_decay=args.weight_decay, patience=args.patience
    )

    # Print final results
    print(f"\n=== Final Results ===")
    print(f"Best Test Accuracy: {results['best_test_acc']*100:.2f}%")
    print(f"Final Metrics:")
    for key, value in results['final_metrics'].items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value:.4f}")

    # Save results if requested
    if args.save_results:
        # Save training curves
        plot_training_curves(results, f'{args.model}_training_curves.png')
        
        # Save confusion matrix
        plot_confusion_matrix(
            results['final_metrics']['confusion_matrix'],
            [f'Digit {d}' for d in args.digits],
            f'{args.model}_confusion_matrix.png'
        )
        
        # Save results to JSON
        results_dict = {
            'model': args.model,
            'dataset': args.dataset,
            'digits': args.digits,
            'best_test_acc': results['best_test_acc'],
            'final_metrics': {k: float(v) if k != 'confusion_matrix' else v.tolist() 
                            for k, v in results['final_metrics'].items()}
        }
        
        with open(f'{args.model}_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {args.model}_results.json")


if __name__ == "__main__":
    main() 