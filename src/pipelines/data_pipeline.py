"""Data pipeline and streaming components for edge learning."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler


class StreamingDataset(Dataset):
    """Dataset for streaming data with edge constraints."""
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        max_samples: int = 1000,
        shuffle: bool = True,
    ) -> None:
        """Initialize streaming dataset.
        
        Args:
            data: Input data array.
            targets: Target labels array.
            max_samples: Maximum samples to keep in memory.
            shuffle: Whether to shuffle the data.
        """
        self.data = data
        self.targets = targets
        self.max_samples = max_samples
        self.shuffle = shuffle
        
        # Create indices
        self.indices = np.arange(len(data))
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        if isinstance(idx, slice):
            indices = self.indices[idx]
            return (
                torch.FloatTensor(self.data[indices]),
                torch.LongTensor(self.targets[indices])
            )
        
        actual_idx = self.indices[idx]
        return (
            torch.FloatTensor(self.data[actual_idx]),
            torch.LongTensor(self.targets[actual_idx])
        )
    
    def get_next_batch(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of data.
        
        Args:
            batch_size: Size of the batch.
            
        Returns:
            Tuple of (data, targets).
        """
        if self.current_idx + batch_size > len(self.data):
            # Reset if we've reached the end
            self.current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        
        end_idx = min(self.current_idx + batch_size, len(self.data))
        batch_indices = self.indices[self.current_idx:end_idx]
        
        self.current_idx = end_idx
        
        return (
            torch.FloatTensor(self.data[batch_indices]),
            torch.LongTensor(self.targets[batch_indices])
        )


class SensorSimulator(ABC):
    """Abstract base class for sensor simulators."""
    
    @abstractmethod
    async def read_data(self) -> np.ndarray:
        """Read data from sensor.
        
        Returns:
            Sensor data array.
        """
        pass
    
    @abstractmethod
    def get_sampling_rate(self) -> float:
        """Get sensor sampling rate.
        
        Returns:
            Sampling rate in Hz.
        """
        pass


class CameraSimulator(SensorSimulator):
    """Simulated camera sensor for image data."""
    
    def __init__(
        self,
        width: int = 28,
        height: int = 28,
        channels: int = 1,
        sampling_rate: float = 30.0,
        noise_level: float = 0.1,
    ) -> None:
        """Initialize camera simulator.
        
        Args:
            width: Image width.
            height: Image height.
            channels: Number of channels.
            sampling_rate: Sampling rate in Hz.
            noise_level: Noise level for simulation.
        """
        self.width = width
        self.height = height
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        
        # Generate synthetic patterns
        self.patterns = self._generate_patterns()
        self.pattern_idx = 0
        
    def _generate_patterns(self) -> List[np.ndarray]:
        """Generate synthetic image patterns."""
        patterns = []
        
        # Simple geometric patterns
        for i in range(10):
            pattern = np.zeros((self.height, self.width))
            
            if i < 3:  # Circles
                center_x, center_y = self.width // 2, self.height // 2
                radius = 8 + i * 2
                y, x = np.ogrid[:self.height, :self.width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                pattern[mask] = 1.0
                
            elif i < 6:  # Rectangles
                start_x, start_y = 4 + i, 4 + i
                end_x, end_y = self.width - 4 - i, self.height - 4 - i
                pattern[start_y:end_y, start_x:end_x] = 1.0
                
            else:  # Lines
                if i == 6:  # Horizontal
                    pattern[self.height//2, :] = 1.0
                elif i == 7:  # Vertical
                    pattern[:, self.width//2] = 1.0
                else:  # Diagonal
                    for j in range(min(self.width, self.height)):
                        pattern[j, j] = 1.0
            
            patterns.append(pattern)
        
        return patterns
    
    async def read_data(self) -> np.ndarray:
        """Read image data from camera."""
        # Simulate processing delay
        await asyncio.sleep(1.0 / self.sampling_rate)
        
        # Get current pattern
        pattern = self.patterns[self.pattern_idx % len(self.patterns)]
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, pattern.shape)
        image = np.clip(pattern + noise, 0, 1)
        
        # Cycle through patterns
        self.pattern_idx += 1
        
        return image.reshape(1, self.height, self.width)
    
    def get_sampling_rate(self) -> float:
        """Get camera sampling rate."""
        return self.sampling_rate


class IMUSimulator(SensorSimulator):
    """Simulated IMU sensor for motion data."""
    
    def __init__(
        self,
        sampling_rate: float = 100.0,
        noise_level: float = 0.05,
    ) -> None:
        """Initialize IMU simulator.
        
        Args:
            sampling_rate: Sampling rate in Hz.
            noise_level: Noise level for simulation.
        """
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        
        # Generate synthetic motion patterns
        self.motion_patterns = self._generate_motion_patterns()
        self.pattern_idx = 0
        self.time_step = 0
        
    def _generate_motion_patterns(self) -> List[np.ndarray]:
        """Generate synthetic motion patterns."""
        patterns = []
        
        # Different motion types
        motion_types = [
            "stationary", "walking", "running", "jumping", "falling",
            "turning_left", "turning_right", "ascending", "descending", "shaking"
        ]
        
        for motion_type in motion_types:
            pattern = np.zeros((100, 6))  # 6-axis IMU data
            
            if motion_type == "stationary":
                # Small random variations
                pattern[:, :] = np.random.normal(0, 0.01, (100, 6))
                
            elif motion_type == "walking":
                # Periodic motion
                t = np.linspace(0, 4*np.pi, 100)
                pattern[:, 0] = 0.5 * np.sin(t)  # X acceleration
                pattern[:, 1] = 0.3 * np.cos(t)  # Y acceleration
                pattern[:, 2] = 9.8 + 0.2 * np.sin(2*t)  # Z acceleration (gravity + variation)
                
            elif motion_type == "running":
                # Higher frequency, larger amplitude
                t = np.linspace(0, 8*np.pi, 100)
                pattern[:, 0] = 1.0 * np.sin(t)
                pattern[:, 1] = 0.5 * np.cos(t)
                pattern[:, 2] = 9.8 + 0.5 * np.sin(4*t)
                
            elif motion_type == "jumping":
                # Impulse-like pattern
                pattern[20:30, 2] = 15.0  # Upward acceleration
                pattern[70:80, 2] = 5.0   # Downward acceleration
                
            elif motion_type == "falling":
                # Free fall pattern
                pattern[:, 2] = 0.0  # No gravity
                pattern[50:, :3] = np.random.normal(0, 2.0, (50, 3))  # Random rotation
                
            patterns.append(pattern)
        
        return patterns
    
    async def read_data(self) -> np.ndarray:
        """Read IMU data."""
        await asyncio.sleep(1.0 / self.sampling_rate)
        
        pattern = self.motion_patterns[self.pattern_idx % len(self.motion_patterns)]
        data_point = pattern[self.time_step % len(pattern)]
        
        # Add noise
        noise = np.random.normal(0, self.noise_level, data_point.shape)
        imu_data = data_point + noise
        
        self.time_step += 1
        
        # Change pattern occasionally
        if self.time_step % 100 == 0:
            self.pattern_idx += 1
        
        return imu_data
    
    def get_sampling_rate(self) -> float:
        """Get IMU sampling rate."""
        return self.sampling_rate


class DataStreamer:
    """Data streaming coordinator for edge learning."""
    
    def __init__(
        self,
        sensor: SensorSimulator,
        buffer_size: int = 100,
        batch_size: int = 1,
    ) -> None:
        """Initialize data streamer.
        
        Args:
            sensor: Sensor simulator.
            buffer_size: Size of the data buffer.
            batch_size: Batch size for processing.
        """
        self.sensor = sensor
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Data buffer
        self.data_buffer: deque = deque(maxlen=buffer_size)
        self.label_buffer: deque = deque(maxlen=buffer_size)
        
        # Streaming state
        self.is_streaming = False
        self.sample_count = 0
        
        # Label generator (simple pattern-based)
        self.label_generator = self._create_label_generator()
        
    def _create_label_generator(self) -> Generator[int, None, None]:
        """Create label generator for streaming data."""
        while True:
            # Generate labels based on sensor patterns
            if hasattr(self.sensor, 'pattern_idx'):
                yield self.sensor.pattern_idx % 10
            else:
                yield np.random.randint(0, 10)
    
    async def start_streaming(self) -> None:
        """Start data streaming."""
        self.is_streaming = True
        logging.info("Started data streaming")
        
        while self.is_streaming:
            try:
                # Read data from sensor
                data = await self.sensor.read_data()
                
                # Generate label
                label = next(self.label_generator)
                
                # Add to buffer
                self.data_buffer.append(data)
                self.label_buffer.append(label)
                
                self.sample_count += 1
                
                # Log progress
                if self.sample_count % 100 == 0:
                    logging.info(f"Streamed {self.sample_count} samples")
                
            except Exception as e:
                logging.error(f"Error in streaming: {e}")
                await asyncio.sleep(0.1)
    
    def stop_streaming(self) -> None:
        """Stop data streaming."""
        self.is_streaming = False
        logging.info("Stopped data streaming")
    
    def get_latest_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get latest batch from buffer.
        
        Args:
            batch_size: Batch size. If None, use default.
            
        Returns:
            Tuple of (data, labels).
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Get latest samples
        latest_data = list(self.data_buffer)[-batch_size:]
        latest_labels = list(self.label_buffer)[-batch_size:]
        
        if not latest_data:
            # Return empty tensors if no data
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        
        # Convert to tensors
        data_tensor = torch.stack([torch.FloatTensor(d) for d in latest_data])
        label_tensor = torch.LongTensor(latest_labels)
        
        return data_tensor, label_tensor
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics.
        """
        return {
            "buffer_size": len(self.data_buffer),
            "max_buffer_size": self.buffer_size,
            "total_samples": self.sample_count,
            "sampling_rate": int(self.sensor.get_sampling_rate()),
        }


def create_synthetic_dataset(
    dataset_type: str = "digits",
    n_samples: int = 1000,
    n_features: int = 64,
    n_classes: int = 10,
    noise: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic dataset for testing.
    
    Args:
        dataset_type: Type of dataset ('digits', 'classification').
        n_samples: Number of samples.
        n_features: Number of features.
        n_classes: Number of classes.
        noise: Noise level.
        
    Returns:
        Tuple of (data, targets).
    """
    if dataset_type == "digits":
        # Use sklearn digits dataset
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    
    elif dataset_type == "classification":
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=n_features,
            random_state=42
        )
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_streaming_dataloader(
    dataset: StreamingDataset,
    batch_size: int = 1,
    shuffle: bool = False,
) -> DataLoader:
    """Create dataloader for streaming data.
    
    Args:
        dataset: Streaming dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        
    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Edge constraint: no multiprocessing
        pin_memory=False,  # Edge constraint: no pinned memory
    )
