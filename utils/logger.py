"""
Logging utilities for quantum text generation
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import torch

class Logger:
    """Simple logger for training and evaluation"""
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 experiment_name: str = "quantum_text_gen",
                 level: int = logging.INFO):
        
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = {}
        self.step_metrics = {}
        
        self.log(f"Logger initialized. Log file: {log_file}")
    
    def log(self, message: str, level: int = logging.INFO):
        """Log a message"""
        self.logger.log(level, message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar metric"""
        if name not in self.step_metrics:
            self.step_metrics[name] = []
        
        self.step_metrics[name].append({
            'step': step,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_dict(self, metrics_dict: Dict[str, float], step: int):
        """Log multiple metrics"""
        for name, value in metrics_dict.items():
            self.log_scalar(name, value, step)
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log(f"Model: {model.__class__.__name__}")
        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Trainable parameters: {trainable_params:,}")
        self.log(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        self.log("Configuration:")
        for key, value in config.items():
            self.log(f"  {key}: {value}")
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{self.experiment_name}_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'step_metrics': self.step_metrics
            }, f, indent=2)
        
        self.log(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.metrics = data.get('metrics', {})
            self.step_metrics = data.get('step_metrics', {})
        
        self.log(f"Metrics loaded from {filepath}")

class ExperimentTracker:
    """Track experiments and results"""
    
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        self.experiments_file = os.path.join(base_dir, "experiments.json")
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> Dict[str, Any]:
        """Load experiment history"""
        if os.path.exists(self.experiments_file):
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experiments(self):
        """Save experiment history"""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def start_experiment(self, 
                        experiment_name: str,
                        config: Dict[str, Any],
                        model_type: str) -> str:
        """Start a new experiment"""
        timestamp = datetime.now().isoformat()
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        exp_dir = os.path.join(self.base_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config
        config_file = os.path.join(exp_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Record experiment
        self.experiments[experiment_id] = {
            'name': experiment_name,
            'model_type': model_type,
            'start_time': timestamp,
            'status': 'running',
            'config': config,
            'directory': exp_dir,
            'results': {}
        }
        
        self._save_experiments()
        return experiment_id
    
    def finish_experiment(self, 
                         experiment_id: str,
                         results: Dict[str, Any],
                         status: str = 'completed'):
        """Finish an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['end_time'] = datetime.now().isoformat()
            self.experiments[experiment_id]['status'] = status
            self.experiments[experiment_id]['results'] = results
            self._save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment information"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, 
                        model_type: Optional[str] = None,
                        status: Optional[str] = None) -> Dict[str, Any]:
        """List experiments with optional filtering"""
        filtered = {}
        for exp_id, exp_data in self.experiments.items():
            if model_type and exp_data.get('model_type') != model_type:
                continue
            if status and exp_data.get('status') != status:
                continue
            filtered[exp_id] = exp_data
        
        return filtered
    
    def get_best_experiment(self, 
                           metric: str = 'val_loss',
                           minimize: bool = True,
                           model_type: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get best experiment based on a metric"""
        best_exp_id = None
        best_value = float('inf') if minimize else float('-inf')
        
        for exp_id, exp_data in self.experiments.items():
            if model_type and exp_data.get('model_type') != model_type:
                continue
            
            if exp_data.get('status') != 'completed':
                continue
            
            results = exp_data.get('results', {})
            if metric not in results:
                continue
            
            value = results[metric]
            if minimize and value < best_value:
                best_value = value
                best_exp_id = exp_id
            elif not minimize and value > best_value:
                best_value = value
                best_exp_id = exp_id
        
        if best_exp_id:
            return best_exp_id, self.experiments[best_exp_id]
        return None

def setup_logging(log_dir: str = "./logs", 
                 experiment_name: str = "quantum_text_gen",
                 level: int = logging.INFO) -> Logger:
    """Setup logging for the experiment"""
    return Logger(log_dir, experiment_name, level)