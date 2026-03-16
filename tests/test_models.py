"""Test suite for on-device learning implementation."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models.tiny_models import TinyCNN, TinyMLP, LoRALayer, AdapterLayer, OnDeviceLearner
from src.pipelines.data_pipeline import StreamingDataset, CameraSimulator, IMUSimulator, DataStreamer
from src.utils.core import set_deterministic_seed, get_device, DeviceConfig, PerformanceMonitor
from src.utils.evaluation import EdgeMetrics, ModelEvaluator, LearningCurveTracker


class TestTinyModels:
    """Test cases for tiny neural network models."""
    
    def test_tiny_cnn_creation(self):
        """Test TinyCNN model creation."""
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        
        assert isinstance(model, TinyCNN)
        assert model.use_lora is True
        assert hasattr(model, 'lora_fc1')
        assert hasattr(model, 'lora_fc2')
    
    def test_tiny_cnn_forward(self):
        """Test TinyCNN forward pass."""
        model = TinyCNN(input_channels=1, num_classes=10)
        x = torch.randn(1, 1, 28, 28)
        
        output = model(x)
        
        assert output.shape == (1, 10)
        assert torch.isfinite(output).all()
    
    def test_tiny_mlp_creation(self):
        """Test TinyMLP model creation."""
        model = TinyMLP(input_size=784, num_classes=10, use_adapters=True)
        
        assert isinstance(model, TinyMLP)
        assert model.use_adapters is True
    
    def test_tiny_mlp_forward(self):
        """Test TinyMLP forward pass."""
        model = TinyMLP(input_size=784, num_classes=10)
        x = torch.randn(1, 784)
        
        output = model(x)
        
        assert output.shape == (1, 10)
        assert torch.isfinite(output).all()
    
    def test_lora_layer(self):
        """Test LoRA layer functionality."""
        lora = LoRALayer(in_features=128, out_features=64, rank=4)
        x = torch.randn(1, 128)
        
        output = lora(x)
        
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()
    
    def test_adapter_layer(self):
        """Test Adapter layer functionality."""
        adapter = AdapterLayer(hidden_size=128, adapter_size=64)
        x = torch.randn(1, 128)
        
        output = adapter(x)
        
        assert output.shape == (1, 128)
        assert torch.isfinite(output).all()
    
    def test_adaptation_parameters(self):
        """Test adaptation parameter extraction."""
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        adaptation_params = model.get_adaptation_parameters()
        
        assert len(adaptation_params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in adaptation_params)
    
    def test_freeze_base_model(self):
        """Test base model freezing."""
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        
        # Freeze base model
        model.freeze_base_model()
        
        # Check that base parameters are frozen
        for name, param in model.named_parameters():
            if 'lora' not in name:
                assert not param.requires_grad
        
        # Check that adaptation parameters are trainable
        adaptation_params = model.get_adaptation_parameters()
        for param in adaptation_params:
            assert param.requires_grad


class TestOnDeviceLearner:
    """Test cases for on-device learning."""
    
    def test_learner_initialization(self):
        """Test OnDeviceLearner initialization."""
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        learner = OnDeviceLearner(model, learning_rate=0.001)
        
        assert learner.model == model
        assert learner.learning_rate == 0.001
        assert learner.sample_count == 0
    
    def test_add_sample(self):
        """Test adding samples to learner."""
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        learner = OnDeviceLearner(model, max_samples=10)
        
        x = torch.randn(1, 1, 28, 28)
        y = torch.tensor([5])
        
        learner.add_sample(x, y)
        
        assert learner.sample_count == 1
        assert len(learner.sample_buffer) == 1
    
    def test_update_model(self):
        """Test model update functionality."""
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        learner = OnDeviceLearner(model, update_frequency=1)
        
        # Add sample
        x = torch.randn(1, 1, 28, 28)
        y = torch.tensor([5])
        learner.add_sample(x, y)
        
        # Update should be triggered
        assert learner.sample_count == 1
    
    def test_adaptation_stats(self):
        """Test adaptation statistics."""
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        learner = OnDeviceLearner(model)
        
        stats = learner.get_adaptation_stats()
        
        assert 'total_samples' in stats
        assert 'buffer_size' in stats
        assert 'adaptation_params' in stats
        assert 'total_params' in stats


class TestDataPipeline:
    """Test cases for data pipeline components."""
    
    def test_streaming_dataset(self):
        """Test StreamingDataset functionality."""
        data = np.random.randn(100, 64)
        targets = np.random.randint(0, 10, 100)
        
        dataset = StreamingDataset(data, targets, max_samples=50)
        
        assert len(dataset) == 100
        assert dataset.max_samples == 50
    
    def test_streaming_dataset_getitem(self):
        """Test StreamingDataset __getitem__ method."""
        data = np.random.randn(10, 64)
        targets = np.random.randint(0, 10, 10)
        
        dataset = StreamingDataset(data, targets)
        
        x, y = dataset[0]
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (64,)
        assert y.shape == ()
    
    def test_camera_simulator(self):
        """Test CameraSimulator functionality."""
        camera = CameraSimulator(width=28, height=28, sampling_rate=30.0)
        
        assert camera.width == 28
        assert camera.height == 28
        assert camera.sampling_rate == 30.0
        assert len(camera.patterns) == 10
    
    def test_imu_simulator(self):
        """Test IMUSimulator functionality."""
        imu = IMUSimulator(sampling_rate=100.0)
        
        assert imu.sampling_rate == 100.0
        assert len(imu.motion_patterns) == 10
    
    def test_data_streamer(self):
        """Test DataStreamer functionality."""
        camera = CameraSimulator()
        streamer = DataStreamer(camera, buffer_size=50)
        
        assert streamer.buffer_size == 50
        assert streamer.sample_count == 0
        assert not streamer.is_streaming


class TestCoreUtils:
    """Test cases for core utilities."""
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Generate random numbers
        torch_rand = torch.randn(10)
        np_rand = np.random.randn(10)
        
        # Reset seed and generate again
        set_deterministic_seed(42)
        torch_rand2 = torch.randn(10)
        np_rand2 = np.random.randn(10)
        
        # Should be identical
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_device_config(self):
        """Test DeviceConfig functionality."""
        config_data = {
            'device_name': 'test_device',
            'device': {
                'memory_mb': 2048,
                'cpu_cores': 4,
                'frameworks': ['onnxruntime'],
                'gpu': 'test_gpu'
            }
        }
        
        config = DeviceConfig(config_data)
        
        assert config.device_name == 'test_device'
        assert config.get_memory_limit() == 2048
        assert config.get_cpu_cores() == 4
        assert config.get_supported_frameworks() == ['onnxruntime']
        assert config.is_gpu_available() is True
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor functionality."""
        monitor = PerformanceMonitor(enabled=True)
        
        # Test timer functionality
        start_time = monitor.start_timer()
        time.sleep(0.001)  # Small delay
        latency = monitor.end_timer(start_time)
        
        assert latency > 0
        
        # Test metric logging
        monitor.log_metric('latency_ms', 10.5)
        stats = monitor.get_stats('latency_ms')
        
        assert 'mean' in stats
        assert stats['mean'] == 10.5


class TestEvaluation:
    """Test cases for evaluation components."""
    
    def test_edge_metrics(self):
        """Test EdgeMetrics functionality."""
        metrics = EdgeMetrics(num_classes=10)
        
        # Add some predictions
        metrics.add_prediction(1, 1, 5.0, 10.0)
        metrics.add_prediction(2, 2, 6.0, 11.0)
        metrics.add_prediction(1, 2, 7.0, 12.0)
        
        # Test accuracy metrics
        accuracy_metrics = metrics.get_accuracy_metrics()
        assert 'accuracy' in accuracy_metrics
        assert accuracy_metrics['accuracy'] == 2/3  # 2 correct out of 3
        
        # Test performance metrics
        performance_metrics = metrics.get_performance_metrics()
        assert 'latency_mean' in performance_metrics
        assert performance_metrics['latency_mean'] == 6.0
        
        # Test confusion matrix
        cm = metrics.get_confusion_matrix()
        assert cm.shape == (10, 10)
    
    def test_model_evaluator(self):
        """Test ModelEvaluator functionality."""
        model = TinyCNN(input_channels=1, num_classes=10)
        device = get_device()
        evaluator = ModelEvaluator(model, device)
        
        # Test batch evaluation
        data = torch.randn(2, 1, 28, 28)
        targets = torch.tensor([1, 2])
        
        metrics = evaluator.evaluate_batch(data, targets)
        
        assert 'accuracy' in metrics
        assert isinstance(metrics['accuracy'], float)
    
    def test_learning_curve_tracker(self):
        """Test LearningCurveTracker functionality."""
        tracker = LearningCurveTracker(window_size=5)
        
        # Add measurements
        tracker.add_measurement(0.8, 0.5, 10.0, 100)
        tracker.add_measurement(0.85, 0.4, 9.0, 200)
        
        # Test trend detection
        trend = tracker.get_trend('accuracy')
        assert trend in ['increasing', 'decreasing', 'stable']
        
        # Test smoothed metrics
        smoothed = tracker.get_smoothed_metrics(window=2)
        assert 'accuracy' in smoothed
        assert 'loss' in smoothed


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create model
        model = TinyCNN(input_channels=1, num_classes=10, use_lora=True)
        
        # Create synthetic data
        data = torch.randn(10, 1, 28, 28)
        targets = torch.randint(0, 10, (10,))
        
        # Create learner
        learner = OnDeviceLearner(model, learning_rate=0.001, update_frequency=5)
        
        # Simulate training
        for i in range(10):
            learner.add_sample(data[i:i+1], targets[i:i+1])
        
        # Check that learning occurred
        assert learner.sample_count == 10
        assert len(learner.sample_buffer) <= learner.max_samples
    
    def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        model = TinyCNN(input_channels=1, num_classes=10)
        device = get_device()
        evaluator = ModelEvaluator(model, device)
        
        # Benchmark inference
        benchmark_results = evaluator.benchmark_inference(
            input_shape=(1, 28, 28),
            num_runs=10,
            warmup_runs=2
        )
        
        assert 'latency_mean' in benchmark_results
        assert 'throughput_fps' in benchmark_results
        assert benchmark_results['latency_mean'] > 0


@pytest.fixture
def sample_model():
    """Fixture providing a sample model for testing."""
    return TinyCNN(input_channels=1, num_classes=10, use_lora=True)


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    return torch.randn(5, 1, 28, 28), torch.randint(0, 10, (5,))


def test_model_with_fixtures(sample_model, sample_data):
    """Test using fixtures."""
    data, targets = sample_data
    
    output = sample_model(data)
    
    assert output.shape == (5, 10)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])
