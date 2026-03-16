"""Core utilities for deterministic behavior and device management."""

import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seed for all random number generators.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device string ('cpu', 'cuda', 'mps'). If None, auto-detect.
        
    Returns:
        PyTorch device object.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configuration objects.
    
    Args:
        base_config: Base configuration.
        override_config: Configuration to override with.
        
    Returns:
        Merged configuration.
    """
    return OmegaConf.merge(base_config, override_config)


class DeviceConfig:
    """Device configuration manager for edge targets."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize device configuration.
        
        Args:
            config: Device configuration dictionary.
        """
        self.config = config
        self.device_name = config.get("device_name", "cpu")
        self.device_specs = config.get("device", {})
        
    def get_memory_limit(self) -> int:
        """Get memory limit in MB.
        
        Returns:
            Memory limit in megabytes.
        """
        return self.device_specs.get("memory_mb", 1024)
    
    def get_cpu_cores(self) -> int:
        """Get number of CPU cores.
        
        Returns:
            Number of CPU cores.
        """
        return self.device_specs.get("cpu_cores", 1)
    
    def get_supported_frameworks(self) -> list[str]:
        """Get supported ML frameworks.
        
        Returns:
            List of supported framework names.
        """
        return self.device_specs.get("frameworks", ["onnxruntime"])
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for this device.
        
        Returns:
            True if GPU is available.
        """
        return "gpu" in self.device_specs and self.device_specs["gpu"] is not None


class PerformanceMonitor:
    """Performance monitoring for edge inference."""
    
    def __init__(self, enabled: bool = True) -> None:
        """Initialize performance monitor.
        
        Args:
            enabled: Whether monitoring is enabled.
        """
        self.enabled = enabled
        self.metrics: Dict[str, list] = {
            "latency_ms": [],
            "memory_mb": [],
            "cpu_usage": [],
        }
        
    def start_timer(self) -> float:
        """Start timing measurement.
        
        Returns:
            Current timestamp.
        """
        if not self.enabled:
            return 0.0
        return torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else time.time()
    
    def end_timer(self, start_time: float) -> float:
        """End timing measurement.
        
        Args:
            start_time: Start timestamp.
            
        Returns:
            Elapsed time in milliseconds.
        """
        if not self.enabled:
            return 0.0
            
        if torch.cuda.is_available():
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            return start_time.elapsed_time(end_event)
        else:
            return (time.time() - start_time) * 1000
    
    def log_metric(self, metric_name: str, value: float) -> None:
        """Log a performance metric.
        
        Args:
            metric_name: Name of the metric.
            value: Metric value.
        """
        if self.enabled and metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric.
            
        Returns:
            Dictionary with mean, std, min, max values.
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            
        values = self.metrics[metric_name]
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            self.metrics[metric].clear()


# Import time for performance monitoring
import time
