"""Main training script for on-device learning implementation."""

import argparse
import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig, OmegaConf

from src.models.tiny_models import TinyCNN, TinyMLP, OnDeviceLearner
from src.pipelines.data_pipeline import (
    create_synthetic_dataset,
    create_streaming_dataloader,
    StreamingDataset,
)
from src.utils.core import set_deterministic_seed, get_device, load_config
from src.utils.evaluation import ModelEvaluator, LearningCurveTracker


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )


def create_model(config: DictConfig) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        Neural network model.
    """
    model_config = config.model
    
    if model_config.architecture == "tiny_cnn":
        model = TinyCNN(
            input_channels=model_config.input_shape[0],
            num_classes=model_config.num_classes,
            use_lora=model_config.on_device_learning.enabled,
            lora_rank=4,
        )
    elif model_config.architecture == "tiny_mlp":
        input_size = model_config.input_shape[0] * model_config.input_shape[1] * model_config.input_shape[2]
        model = TinyMLP(
            input_size=input_size,
            num_classes=model_config.num_classes,
            use_adapters=model_config.on_device_learning.enabled,
        )
    else:
        raise ValueError(f"Unknown architecture: {model_config.architecture}")
    
    return model


def prepare_data(config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare training, validation, and test data.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create synthetic dataset
    X, y = create_synthetic_dataset(
        dataset_type="digits",
        n_samples=config.data.get("n_samples", 2000),
        n_classes=config.model.num_classes,
    )
    
    # Reshape for CNN if needed
    if config.model.architecture == "tiny_cnn":
        X = X.reshape(-1, *config.model.input_shape)
    
    # Create dataset
    dataset = StreamingDataset(X, y, max_samples=config.data.get("max_samples", 1000))
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = config.runtime.inference.batch_size
    
    train_loader = create_streaming_dataloader(train_dataset, batch_size=batch_size)
    val_loader = create_streaming_dataloader(val_dataset, batch_size=batch_size)
    test_loader = create_streaming_dataloader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DictConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Train the model with on-device learning capabilities.
    
    Args:
        model: Neural network model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Configuration object.
        device: Device for computation.
        
    Returns:
        Dictionary with training metrics.
    """
    model = model.to(device)
    
    # Setup optimizer
    if config.model.on_device_learning.enabled:
        # Only train adaptation parameters
        if hasattr(model, 'get_adaptation_parameters'):
            adaptation_params = model.get_adaptation_parameters()
        elif hasattr(model, 'get_adapter_parameters'):
            adaptation_params = model.get_adapter_parameters()
        else:
            adaptation_params = list(model.parameters())
        
        optimizer = optim.Adam(
            adaptation_params,
            lr=config.model.on_device_learning.learning_rate,
        )
    else:
        # Train all parameters
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.model.on_device_learning.learning_rate,
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize on-device learner
    on_device_learner = OnDeviceLearner(
        model=model,
        learning_rate=config.model.on_device_learning.learning_rate,
        batch_size=config.model.on_device_learning.batch_size,
        max_samples=config.model.on_device_learning.max_samples_per_update,
        update_frequency=config.model.on_device_learning.update_frequency,
    )
    
    # Learning curve tracker
    curve_tracker = LearningCurveTracker()
    
    # Training loop
    model.train()
    best_val_accuracy = 0.0
    training_metrics = {
        "train_loss": [],
        "val_accuracy": [],
        "learning_samples": [],
    }
    
    logging.info("Starting training...")
    
    for epoch in range(config.training.get("epochs", 10)):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # On-device learning updates
            if config.model.on_device_learning.enabled:
                for i in range(len(data)):
                    on_device_learner.add_sample(
                        data[i:i+1], targets[i:i+1]
                    )
            
            # Log progress
            if batch_idx % 100 == 0:
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        
        # Validation
        model.eval()
        val_accuracy = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                val_accuracy += (predictions == targets).sum().item()
                val_samples += len(targets)
        
        val_accuracy /= val_samples
        
        # Track learning curve
        curve_tracker.add_measurement(
            accuracy=val_accuracy,
            loss=avg_loss,
            latency=0.0,  # Will be measured separately
            sample_count=on_device_learner.sample_count,
        )
        
        # Store metrics
        training_metrics["train_loss"].append(avg_loss)
        training_metrics["val_accuracy"].append(val_accuracy)
        training_metrics["learning_samples"].append(on_device_learner.sample_count)
        
        # Log epoch results
        logging.info(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Accuracy={val_accuracy:.4f}, "
            f"Samples={on_device_learner.sample_count}"
        )
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
        
        model.train()
    
    # Final metrics
    final_metrics = {
        "best_val_accuracy": best_val_accuracy,
        "final_train_loss": training_metrics["train_loss"][-1],
        "total_samples": on_device_learner.sample_count,
        "adaptation_stats": on_device_learner.get_adaptation_stats(),
    }
    
    return final_metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: DictConfig,
) -> Dict[str, float]:
    """Evaluate the trained model.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device for computation.
        config: Configuration object.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    # Load best model
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
    
    model = model.to(device)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, config.model.num_classes)
    
    # Evaluate on test set
    logging.info("Evaluating model on test set...")
    test_metrics = evaluator.evaluate_dataset(test_loader)
    
    # Benchmark inference
    logging.info("Benchmarking inference performance...")
    benchmark_metrics = evaluator.benchmark_inference(
        input_shape=config.model.input_shape,
        num_runs=100,
    )
    
    # Get model size
    model_size = evaluator.get_model_size()
    
    # Combine all metrics
    all_metrics = {
        "test_metrics": test_metrics,
        "benchmark_metrics": benchmark_metrics,
        "model_size_mb": model_size,
    }
    
    return all_metrics


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="On-Device Learning Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/device/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_deterministic_seed(args.seed)
    device = get_device(args.device)
    
    # Load configuration
    config = load_config(args.config)
    
    logging.info(f"Using device: {device}")
    logging.info(f"Configuration: {config}")
    
    # Create model
    model = create_model(config)
    logging.info(f"Created model: {model.__class__.__name__}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    logging.info(f"Data prepared: {len(train_loader)} train batches, {len(val_loader)} val batches, {len(test_loader)} test batches")
    
    # Train model
    training_metrics = train_model(model, train_loader, val_loader, config, device)
    logging.info(f"Training completed. Best validation accuracy: {training_metrics['best_val_accuracy']:.4f}")
    
    # Evaluate model
    evaluation_metrics = evaluate_model(model, test_loader, device, config)
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Best Validation Accuracy: {training_metrics['best_val_accuracy']:.4f}")
    print(f"Total Learning Samples: {training_metrics['total_samples']}")
    print(f"Model Size: {evaluation_metrics['model_size_mb']:.2f} MB")
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    test_metrics = evaluation_metrics['test_metrics']
    print(f"Test Accuracy: {test_metrics['accuracy']['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['accuracy']['f1_score']:.4f}")
    
    benchmark_metrics = evaluation_metrics['benchmark_metrics']
    print(f"Mean Latency: {benchmark_metrics['latency_mean']:.2f} ms")
    print(f"P95 Latency: {benchmark_metrics['latency_p95']:.2f} ms")
    print(f"Throughput: {benchmark_metrics['throughput_fps']:.2f} FPS")
    
    print("\n" + "="*50)
    print("ON-DEVICE LEARNING STATS")
    print("="*50)
    adaptation_stats = training_metrics['adaptation_stats']
    print(f"Adaptation Parameters: {adaptation_stats['adaptation_params']}")
    print(f"Total Parameters: {adaptation_stats['total_params']}")
    print(f"Parameter Efficiency: {adaptation_stats['adaptation_params'] / adaptation_stats['total_params']:.4f}")


if __name__ == "__main__":
    main()
