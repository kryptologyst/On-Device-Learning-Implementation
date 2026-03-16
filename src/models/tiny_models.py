"""Lightweight neural network models for on-device learning."""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.
    
    LoRA reduces the number of trainable parameters by decomposing weight updates
    into low-rank matrices, making it ideal for on-device learning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            rank: Rank of the low-rank decomposition.
            alpha: Scaling factor for LoRA updates.
            dropout: Dropout probability.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # LoRA computation: x @ A^T @ B^T
        x = self.dropout(x)
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.alpha / self.rank


class TinyCNN(nn.Module):
    """Tiny Convolutional Neural Network for edge devices.
    
    A lightweight CNN designed for on-device learning with minimal memory footprint.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        use_lora: bool = True,
        lora_rank: int = 4,
    ) -> None:
        """Initialize TinyCNN.
        
        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
            use_lora: Whether to use LoRA for adaptation.
            lora_rank: Rank for LoRA layers.
        """
        super().__init__()
        self.use_lora = use_lora
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate flattened size (assuming 28x28 input)
        self.flattened_size = 64 * 3 * 3  # After 3 pooling operations
        
        # Classification head
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # LoRA layers for adaptation
        if use_lora:
            self.lora_fc1 = LoRALayer(128, num_classes, rank=lora_rank)
            self.lora_fc2 = LoRALayer(self.flattened_size, 128, rank=lora_rank)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        if self.use_lora:
            # Add LoRA adaptation
            lora_output = self.lora_fc2(x.view(x.size(0), -1))
            x = x + lora_output
        
        x = self.fc2(x)
        
        if self.use_lora:
            # Add LoRA adaptation to final layer
            lora_output = self.lora_fc1(x)
            x = x + lora_output
        
        return x
    
    def get_adaptation_parameters(self) -> List[Parameter]:
        """Get parameters used for adaptation (LoRA layers).
        
        Returns:
            List of adaptation parameters.
        """
        if self.use_lora:
            return list(self.lora_fc1.parameters()) + list(self.lora_fc2.parameters())
        return []
    
    def freeze_base_model(self) -> None:
        """Freeze the base model parameters, keeping only adaptation layers trainable."""
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze adaptation parameters
        for param in self.get_adaptation_parameters():
            param.requires_grad = True


class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning.
    
    Adapters add small bottleneck layers between existing layers,
    enabling efficient adaptation with minimal parameter overhead.
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        """Initialize adapter layer.
        
        Args:
            hidden_size: Size of the hidden dimension.
            adapter_size: Size of the adapter bottleneck.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__()
        self.adapter_size = adapter_size
        
        # Adapter layers
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # Down-project
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Up-project
        x = self.up_proj(x)
        
        return x


class TinyMLP(nn.Module):
    """Tiny Multi-Layer Perceptron for edge devices.
    
    A minimal MLP designed for on-device learning with adapters.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 10,
        num_layers: int = 2,
        use_adapters: bool = True,
        adapter_size: int = 32,
    ) -> None:
        """Initialize TinyMLP.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden layer size.
            num_classes: Number of output classes.
            num_layers: Number of hidden layers.
            use_adapters: Whether to use adapter layers.
            adapter_size: Size of adapter bottleneck.
        """
        super().__init__()
        self.use_adapters = use_adapters
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            
            # Add adapter if enabled
            if use_adapters:
                layers.append(AdapterLayer(hidden_size, adapter_size))
        
        layers.append(nn.Linear(hidden_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        return self.network(x)
    
    def get_adapter_parameters(self) -> List[Parameter]:
        """Get adapter parameters for adaptation.
        
        Returns:
            List of adapter parameters.
        """
        if not self.use_adapters:
            return []
        
        adapter_params = []
        for module in self.network.modules():
            if isinstance(module, AdapterLayer):
                adapter_params.extend(list(module.parameters()))
        return adapter_params


class OnDeviceLearner:
    """On-device learning coordinator for incremental model updates."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 1,
        max_samples: int = 100,
        update_frequency: int = 10,
    ) -> None:
        """Initialize on-device learner.
        
        Args:
            model: Neural network model.
            learning_rate: Learning rate for adaptation.
            batch_size: Batch size for updates (typically 1 for edge).
            max_samples: Maximum samples to keep in memory.
            update_frequency: Update frequency in samples.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.update_frequency = update_frequency
        
        # Get adaptation parameters
        if hasattr(model, 'get_adaptation_parameters'):
            self.adaptation_params = model.get_adaptation_parameters()
        elif hasattr(model, 'get_adapter_parameters'):
            self.adaptation_params = model.get_adapter_parameters()
        else:
            self.adaptation_params = list(model.parameters())
        
        # Optimizer for adaptation
        self.optimizer = torch.optim.Adam(self.adaptation_params, lr=learning_rate)
        
        # Memory buffer for samples
        self.sample_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.sample_count = 0
        
    def add_sample(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a new sample to the learning buffer.
        
        Args:
            x: Input features.
            y: Target labels.
        """
        self.sample_buffer.append((x.clone(), y.clone()))
        self.sample_count += 1
        
        # Maintain buffer size
        if len(self.sample_buffer) > self.max_samples:
            self.sample_buffer.pop(0)
        
        # Update model if frequency reached
        if self.sample_count % self.update_frequency == 0:
            self.update_model()
    
    def update_model(self) -> None:
        """Update the model using buffered samples."""
        if not self.sample_buffer:
            return
        
        self.model.train()
        
        # Create mini-batch
        batch_x = torch.cat([sample[0] for sample in self.sample_buffer[-self.batch_size:]])
        batch_y = torch.cat([sample[1] for sample in self.sample_buffer[-self.batch_size:]])
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()
    
    def get_adaptation_stats(self) -> Dict[str, int]:
        """Get statistics about adaptation.
        
        Returns:
            Dictionary with adaptation statistics.
        """
        return {
            "total_samples": self.sample_count,
            "buffer_size": len(self.sample_buffer),
            "adaptation_params": len(self.adaptation_params),
            "total_params": sum(p.numel() for p in self.model.parameters()),
        }
