"""
Training module for quantum text generation models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import os
import json
from tqdm import tqdm

from models.quantum_text_models import QuantumTextGenerator
from utils.metrics import calculate_perplexity, calculate_bleu_score
from utils.logger import Logger

class QuantumTextTrainer:
    """Trainer for quantum text generation models"""
    
    def __init__(self,
                 model: QuantumTextGenerator,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cpu',
                 logger: Optional[Logger] = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger or Logger()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                if self.model.model_type == 'qrwkv':
                    logits, _ = self.model(input_ids)
                else:
                    logits = self.model(input_ids)
                
                # Calculate loss
                loss = self.calculate_loss(logits, target_ids, attention_mask)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                batch_tokens = attention_mask.sum().item()
                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'ppl': f'{np.exp(batch_loss):.2f}'
                })
                
                self.global_step += 1
                
                # Log training step
                if self.global_step % 100 == 0:
                    self.logger.log_scalar('train/step_loss', batch_loss, self.global_step)
                    self.logger.log_scalar('train/step_perplexity', np.exp(batch_loss), self.global_step)
            
            except Exception as e:
                self.logger.log(f"Error in training step {batch_idx}: {str(e)}")
                continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                try:
                    # Forward pass
                    if self.model.model_type == 'qrwkv':
                        logits, _ = self.model(input_ids)
                    else:
                        logits = self.model(input_ids)
                    
                    # Calculate loss
                    loss = self.calculate_loss(logits, target_ids, attention_mask)
                    
                    # Update metrics
                    batch_loss = loss.item()
                    batch_tokens = attention_mask.sum().item()
                    total_loss += batch_loss * batch_tokens
                    total_tokens += batch_tokens
                
                except Exception as e:
                    self.logger.log(f"Error in validation step: {str(e)}")
                    continue
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens
        }
    
    def calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Calculate language modeling loss"""
        # Reshape logits and targets
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]
        
        # Calculate cross-entropy loss
        loss = self.criterion(logits, targets)
        
        return loss
    
    def train(self, 
              epochs: int,
              save_dir: str = './checkpoints',
              eval_every: int = 5,
              save_every: int = 10,
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """Main training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.log(f"Starting training for {epochs} epochs")
        self.logger.log(f"Model: {self.model.model_type}")
        self.logger.log(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.log(f"Validation samples: {len(self.val_loader.dataset)}")
        
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log training metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_perplexity'].append(train_metrics['perplexity'])
            
            if self.scheduler:
                self.training_history['learning_rate'].append(self.scheduler.get_last_lr()[0])
                self.scheduler.step()
            
            self.logger.log(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train PPL: {train_metrics['perplexity']:.2f}")
            
            # Validation
            if epoch % eval_every == 0 or epoch == epochs - 1:
                val_metrics = self.validate()
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_perplexity'].append(val_metrics['perplexity'])
                
                self.logger.log(f"Epoch {epoch}: Val Loss: {val_metrics['loss']:.4f}, "
                              f"Val PPL: {val_metrics['perplexity']:.2f}")
                
                # Log to tensorboard/wandb if available
                self.logger.log_scalar('train/epoch_loss', train_metrics['loss'], epoch)
                self.logger.log_scalar('train/epoch_perplexity', train_metrics['perplexity'], epoch)
                self.logger.log_scalar('val/epoch_loss', val_metrics['loss'], epoch)
                self.logger.log_scalar('val/epoch_perplexity', val_metrics['perplexity'], epoch)
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    early_stopping_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        os.path.join(save_dir, 'best_model.pth'),
                        epoch, val_metrics['loss'], is_best=True
                    )
                else:
                    early_stopping_counter += 1
                
                # Early stopping
                if early_stopping_counter >= early_stopping_patience:
                    self.logger.log(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'),
                    epoch, train_metrics['loss']
                )
        
        total_time = time.time() - start_time
        self.logger.log(f"Training completed in {total_time:.2f} seconds")
        self.logger.log(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.training_history
    
    def save_checkpoint(self, path: str, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'training_history': self.training_history,
            'model_type': self.model.model_type
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            self.logger.log(f"Saved best model checkpoint: {path}")
        else:
            self.logger.log(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'val_loss': [], 'train_perplexity': [], 
            'val_perplexity': [], 'learning_rate': []
        })
        
        self.logger.log(f"Loaded checkpoint from {path}, epoch {self.epoch}")

def create_optimizer(model: nn.Module, 
                    optimizer_type: str = 'adam',
                    learning_rate: float = 1e-3,
                    weight_decay: float = 1e-5,
                    **kwargs) -> torch.optim.Optimizer:
    """Create optimizer for the model"""
    
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                        momentum=kwargs.get('momentum', 0.9), **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = 'cosine',
                    num_epochs: int = 100,
                    **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler"""
    
    if scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, **kwargs)
    elif scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 30), 
                                        gamma=kwargs.get('gamma', 0.1))
    elif scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=kwargs.get('factor', 0.5),
                                                   patience=kwargs.get('patience', 5))
    elif scheduler_type.lower() == 'linear':
        return optim.lr_scheduler.LinearLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'constant':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")