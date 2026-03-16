"""Evaluation metrics and performance analysis for edge learning."""

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..utils.core import PerformanceMonitor


class EdgeMetrics:
    """Comprehensive metrics for edge AI evaluation."""
    
    def __init__(self, num_classes: int = 10) -> None:
        """Initialize edge metrics.
        
        Args:
            num_classes: Number of classes for classification.
        """
        self.num_classes = num_classes
        self.reset()
        
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions: List[int] = []
        self.targets: List[int] = []
        self.latencies: List[float] = []
        self.memory_usage: List[float] = []
        self.model_sizes: List[float] = []
        
    def add_prediction(
        self,
        prediction: int,
        target: int,
        latency: float,
        memory_usage: float,
    ) -> None:
        """Add a prediction result.
        
        Args:
            prediction: Predicted class.
            target: True class.
            latency: Inference latency in milliseconds.
            memory_usage: Memory usage in MB.
        """
        self.predictions.append(prediction)
        self.targets.append(target)
        self.latencies.append(latency)
        self.memory_usage.append(memory_usage)
    
    def add_model_size(self, size_mb: float) -> None:
        """Add model size measurement.
        
        Args:
            size_mb: Model size in MB.
        """
        self.model_sizes.append(size_mb)
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Get accuracy-related metrics.
        
        Returns:
            Dictionary with accuracy metrics.
        """
        if not self.predictions:
            return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        accuracy = accuracy_score(self.targets, self.predictions)
        f1 = f1_score(self.targets, self.predictions, average="weighted")
        precision = precision_score(self.targets, self.predictions, average="weighted")
        recall = recall_score(self.targets, self.predictions, average="weighted")
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance-related metrics.
        
        Returns:
            Dictionary with performance metrics.
        """
        if not self.latencies:
            return {
                "latency_mean": 0.0,
                "latency_std": 0.0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "throughput_fps": 0.0,
                "memory_mean": 0.0,
                "memory_max": 0.0,
            }
        
        latencies = np.array(self.latencies)
        memory_usage = np.array(self.memory_usage)
        
        return {
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_min": np.min(latencies),
            "latency_max": np.max(latencies),
            "throughput_fps": 1000.0 / np.mean(latencies),  # FPS
            "memory_mean": np.mean(memory_usage),
            "memory_max": np.max(memory_usage),
            "memory_std": np.std(memory_usage),
        }
    
    def get_model_efficiency_metrics(self) -> Dict[str, float]:
        """Get model efficiency metrics.
        
        Returns:
            Dictionary with efficiency metrics.
        """
        if not self.model_sizes:
            return {"model_size_mb": 0.0, "params_per_accuracy": 0.0}
        
        model_size = np.mean(self.model_sizes)
        accuracy_metrics = self.get_accuracy_metrics()
        accuracy = accuracy_metrics["accuracy"]
        
        # Estimate parameters (rough approximation)
        estimated_params = model_size * 1024 * 1024 / 4  # Assuming float32
        
        return {
            "model_size_mb": model_size,
            "estimated_params": estimated_params,
            "params_per_accuracy": estimated_params / max(accuracy, 1e-6),
            "accuracy_per_mb": accuracy / max(model_size, 1e-6),
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.
        
        Returns:
            Confusion matrix array.
        """
        if not self.predictions:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(self.targets, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            Classification report string.
        """
        if not self.predictions:
            return "No predictions available"
        
        return classification_report(
            self.targets,
            self.predictions,
            target_names=[f"Class_{i}" for i in range(self.num_classes)],
        )
    
    def get_all_metrics(self) -> Dict[str, Union[float, np.ndarray, str]]:
        """Get all available metrics.
        
        Returns:
            Dictionary with all metrics.
        """
        return {
            "accuracy": self.get_accuracy_metrics(),
            "performance": self.get_performance_metrics(),
            "efficiency": self.get_model_efficiency_metrics(),
            "confusion_matrix": self.get_confusion_matrix(),
            "classification_report": self.get_classification_report(),
        }


class ModelEvaluator:
    """Model evaluator for edge learning systems."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 10,
    ) -> None:
        """Initialize model evaluator.
        
        Args:
            model: Neural network model.
            device: Device for computation.
            num_classes: Number of classes.
        """
        self.model = model
        self.device = device
        self.metrics = EdgeMetrics(num_classes)
        self.performance_monitor = PerformanceMonitor()
        
    def evaluate_batch(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        measure_performance: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model on a batch of data.
        
        Args:
            data: Input data tensor.
            targets: Target labels tensor.
            measure_performance: Whether to measure performance metrics.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Measure inference time
            if measure_performance:
                start_time = self.performance_monitor.start_timer()
            
            # Forward pass
            outputs = self.model(data)
            predictions = torch.argmax(outputs, dim=1)
            
            if measure_performance:
                latency = self.performance_monitor.end_timer(start_time)
            else:
                latency = 0.0
            
            # Get memory usage (approximate)
            memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
            
            # Add to metrics
            for i in range(len(predictions)):
                self.metrics.add_prediction(
                    prediction=predictions[i].item(),
                    target=targets[i].item(),
                    latency=latency,
                    memory_usage=memory_usage,
                )
        
        return self.metrics.get_accuracy_metrics()
    
    def evaluate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """Evaluate model on entire dataset.
        
        Args:
            dataloader: DataLoader for evaluation data.
            max_batches: Maximum number of batches to evaluate.
            
        Returns:
            Dictionary with all evaluation metrics.
        """
        self.metrics.reset()
        
        batch_count = 0
        for batch_data, batch_targets in dataloader:
            self.evaluate_batch(batch_data, batch_targets)
            batch_count += 1
            
            if max_batches and batch_count >= max_batches:
                break
        
        return self.metrics.get_all_metrics()
    
    def benchmark_inference(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            input_shape: Input tensor shape.
            num_runs: Number of benchmark runs.
            warmup_runs: Number of warmup runs.
            
        Returns:
            Dictionary with benchmark results.
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
        
        # Benchmark runs
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
        
        latencies = np.array(latencies)
        
        return {
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "throughput_fps": 1000.0 / np.mean(latencies),
        }
    
    def get_model_size(self) -> float:
        """Get model size in MB.
        
        Returns:
            Model size in megabytes.
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate size (assuming float32)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)
        
        self.metrics.add_model_size(size_mb)
        return size_mb


class LearningCurveTracker:
    """Track learning progress over time."""
    
    def __init__(self, window_size: int = 100) -> None:
        """Initialize learning curve tracker.
        
        Args:
            window_size: Size of the sliding window for metrics.
        """
        self.window_size = window_size
        self.reset()
        
    def reset(self) -> None:
        """Reset tracking data."""
        self.accuracy_history: List[float] = []
        self.loss_history: List[float] = []
        self.latency_history: List[float] = []
        self.sample_count_history: List[int] = []
        
    def add_measurement(
        self,
        accuracy: float,
        loss: float,
        latency: float,
        sample_count: int,
    ) -> None:
        """Add a measurement point.
        
        Args:
            accuracy: Current accuracy.
            loss: Current loss.
            latency: Current latency.
            sample_count: Number of samples processed.
        """
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.latency_history.append(latency)
        self.sample_count_history.append(sample_count)
        
        # Maintain window size
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)
            self.loss_history.pop(0)
            self.latency_history.pop(0)
            self.sample_count_history.pop(0)
    
    def get_trend(self, metric: str = "accuracy") -> str:
        """Get trend direction for a metric.
        
        Args:
            metric: Metric name ('accuracy', 'loss', 'latency').
            
        Returns:
            Trend direction ('increasing', 'decreasing', 'stable').
        """
        if metric == "accuracy":
            history = self.accuracy_history
        elif metric == "loss":
            history = self.loss_history
        elif metric == "latency":
            history = self.latency_history
        else:
            return "unknown"
        
        if len(history) < 2:
            return "insufficient_data"
        
        # Calculate trend using linear regression
        x = np.arange(len(history))
        y = np.array(history)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_smoothed_metrics(self, window: int = 10) -> Dict[str, List[float]]:
        """Get smoothed metrics using moving average.
        
        Args:
            window: Window size for smoothing.
            
        Returns:
            Dictionary with smoothed metrics.
        """
        if len(self.accuracy_history) < window:
            return {
                "accuracy": self.accuracy_history,
                "loss": self.loss_history,
                "latency": self.latency_history,
            }
        
        def smooth(data: List[float]) -> List[float]:
            smoothed = []
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                smoothed.append(np.mean(data[start_idx:i+1]))
            return smoothed
        
        return {
            "accuracy": smooth(self.accuracy_history),
            "loss": smooth(self.loss_history),
            "latency": smooth(self.latency_history),
        }


def create_evaluation_report(
    metrics: Dict[str, Union[float, np.ndarray, str]],
    model_name: str = "Model",
    device_name: str = "Edge Device",
) -> str:
    """Create a comprehensive evaluation report.
    
    Args:
        metrics: Evaluation metrics dictionary.
        model_name: Name of the model.
        device_name: Name of the device.
        
    Returns:
        Formatted evaluation report.
    """
    report = f"""
=== {model_name} Evaluation Report ===
Device: {device_name}

ACCURACY METRICS:
- Accuracy: {metrics['accuracy']['accuracy']:.4f}
- F1 Score: {metrics['accuracy']['f1_score']:.4f}
- Precision: {metrics['accuracy']['precision']:.4f}
- Recall: {metrics['accuracy']['recall']:.4f}

PERFORMANCE METRICS:
- Mean Latency: {metrics['performance']['latency_mean']:.2f} ms
- P95 Latency: {metrics['performance']['latency_p95']:.2f} ms
- Throughput: {metrics['performance']['throughput_fps']:.2f} FPS
- Mean Memory: {metrics['performance']['memory_mean']:.2f} MB
- Max Memory: {metrics['performance']['memory_max']:.2f} MB

EFFICIENCY METRICS:
- Model Size: {metrics['efficiency']['model_size_mb']:.2f} MB
- Estimated Parameters: {metrics['efficiency']['estimated_params']:.0f}
- Parameters per Accuracy: {metrics['efficiency']['params_per_accuracy']:.0f}
- Accuracy per MB: {metrics['efficiency']['accuracy_per_mb']:.4f}

CLASSIFICATION REPORT:
{metrics['classification_report']}
"""
    return report
