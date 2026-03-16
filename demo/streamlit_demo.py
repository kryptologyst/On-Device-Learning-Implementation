"""Streamlit demo for on-device learning visualization."""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

from src.models.tiny_models import TinyCNN, TinyMLP, OnDeviceLearner
from src.pipelines.data_pipeline import CameraSimulator, IMUSimulator, DataStreamer
from src.utils.core import set_deterministic_seed, get_device
from src.utils.evaluation import EdgeMetrics, LearningCurveTracker


# Page configuration
st.set_page_config(
    page_title="On-Device Learning Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "learner" not in st.session_state:
    st.session_state.learner = None
if "streamer" not in st.session_state:
    st.session_state.streamer = None
if "metrics" not in st.session_state:
    st.session_state.metrics = EdgeMetrics()
if "curve_tracker" not in st.session_state:
    st.session_state.curve_tracker = LearningCurveTracker()
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False
if "streaming_task" not in st.session_state:
    st.session_state.streaming_task = None


def create_model(model_type: str, use_adaptation: bool) -> nn.Module:
    """Create model based on user selection.
    
    Args:
        model_type: Type of model ('CNN', 'MLP').
        use_adaptation: Whether to use adaptation layers.
        
    Returns:
        Neural network model.
    """
    if model_type == "CNN":
        return TinyCNN(
            input_channels=1,
            num_classes=10,
            use_lora=use_adaptation,
            lora_rank=4,
        )
    else:  # MLP
        return TinyMLP(
            input_size=28 * 28,
            num_classes=10,
            use_adapters=use_adaptation,
        )


def simulate_inference(model: nn.Module, data: torch.Tensor) -> Tuple[int, float]:
    """Simulate model inference with timing.
    
    Args:
        model: Neural network model.
        data: Input data tensor.
        
    Returns:
        Tuple of (prediction, latency_ms).
    """
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(data)
        prediction = torch.argmax(outputs, dim=1).item()
    
    latency = (time.time() - start_time) * 1000  # Convert to ms
    return prediction, latency


def create_performance_plot(metrics: EdgeMetrics) -> go.Figure:
    """Create performance visualization.
    
    Args:
        metrics: Edge metrics object.
        
    Returns:
        Plotly figure.
    """
    if not metrics.latencies:
        # Return empty plot
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Latency Over Time", "Memory Usage", "Accuracy Trend", "Throughput"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Latency over time
    fig.add_trace(
        go.Scatter(
            y=metrics.latencies,
            mode="lines",
            name="Latency (ms)",
            line=dict(color="blue"),
        ),
        row=1, col=1
    )
    
    # Memory usage
    fig.add_trace(
        go.Scatter(
            y=metrics.memory_usage,
            mode="lines",
            name="Memory (MB)",
            line=dict(color="red"),
        ),
        row=1, col=2
    )
    
    # Accuracy trend (if available)
    if metrics.predictions and metrics.targets:
        # Calculate rolling accuracy
        window_size = min(50, len(metrics.predictions))
        rolling_accuracy = []
        
        for i in range(len(metrics.predictions)):
            start_idx = max(0, i - window_size + 1)
            recent_preds = metrics.predictions[start_idx:i+1]
            recent_targets = metrics.targets[start_idx:i+1]
            
            if recent_preds:
                accuracy = sum(p == t for p, t in zip(recent_preds, recent_targets)) / len(recent_preds)
                rolling_accuracy.append(accuracy)
        
        fig.add_trace(
            go.Scatter(
                y=rolling_accuracy,
                mode="lines",
                name="Rolling Accuracy",
                line=dict(color="green"),
            ),
            row=2, col=1
        )
    
    # Throughput
    if metrics.latencies:
        throughput = [1000.0 / lat if lat > 0 else 0 for lat in metrics.latencies]
        fig.add_trace(
            go.Scatter(
                y=throughput,
                mode="lines",
                name="Throughput (FPS)",
                line=dict(color="orange"),
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Real-time Performance Metrics"
    )
    
    return fig


def create_learning_curve_plot(curve_tracker: LearningCurveTracker) -> go.Figure:
    """Create learning curve visualization.
    
    Args:
        curve_tracker: Learning curve tracker.
        
    Returns:
        Plotly figure.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy Over Time", "Loss Over Time"),
    )
    
    if curve_tracker.accuracy_history:
        fig.add_trace(
            go.Scatter(
                y=curve_tracker.accuracy_history,
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="blue"),
            ),
            row=1, col=1
        )
    
    if curve_tracker.loss_history:
        fig.add_trace(
            go.Scatter(
                y=curve_tracker.loss_history,
                mode="lines+markers",
                name="Loss",
                line=dict(color="red"),
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Learning Progress"
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 On-Device Learning Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This implementation is NOT intended for safety-critical applications or production deployment. 
    It is designed for learning, experimentation, and showcasing on-device learning concepts.</p>
    <p>Use at your own risk. The authors assume no responsibility for any consequences of using this code.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Architecture",
        ["CNN", "MLP"],
        help="Choose between Convolutional Neural Network or Multi-Layer Perceptron"
    )
    
    use_adaptation = st.sidebar.checkbox(
        "Enable On-Device Learning",
        value=True,
        help="Enable LoRA/Adapter layers for efficient adaptation"
    )
    
    # Sensor selection
    sensor_type = st.sidebar.selectbox(
        "Sensor Type",
        ["Camera", "IMU"],
        help="Choose sensor simulator type"
    )
    
    # Streaming parameters
    st.sidebar.subheader("Streaming Parameters")
    buffer_size = st.sidebar.slider("Buffer Size", 10, 200, 100)
    update_frequency = st.sidebar.slider("Update Frequency", 1, 50, 10)
    sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 1, 100, 30)
    
    # Initialize model if needed
    if st.session_state.model is None:
        with st.spinner("Initializing model..."):
            set_deterministic_seed(42)
            device = get_device()
            st.session_state.model = create_model(model_type, use_adaptation)
            st.session_state.model = st.session_state.model.to(device)
            
            # Initialize learner
            st.session_state.learner = OnDeviceLearner(
                model=st.session_state.model,
                learning_rate=0.001,
                batch_size=1,
                max_samples=buffer_size,
                update_frequency=update_frequency,
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Real-time Performance")
        
        # Performance plot
        performance_fig = create_performance_plot(st.session_state.metrics)
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Learning curve plot
        st.header("Learning Progress")
        learning_fig = create_learning_curve_plot(st.session_state.curve_tracker)
        st.plotly_chart(learning_fig, use_container_width=True)
    
    with col2:
        st.header("Control Panel")
        
        # Start/Stop streaming
        if not st.session_state.is_streaming:
            if st.button("🚀 Start Streaming", type="primary"):
                # Initialize sensor and streamer
                if sensor_type == "Camera":
                    sensor = CameraSimulator(sampling_rate=sampling_rate)
                else:  # IMU
                    sensor = IMUSimulator(sampling_rate=sampling_rate)
                
                st.session_state.streamer = DataStreamer(
                    sensor=sensor,
                    buffer_size=buffer_size,
                    batch_size=1,
                )
                
                st.session_state.is_streaming = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Streaming"):
                st.session_state.is_streaming = False
                if st.session_state.streamer:
                    st.session_state.streamer.stop_streaming()
                st.rerun()
        
        # Model information
        st.subheader("Model Information")
        if st.session_state.model:
            total_params = sum(p.numel() for p in st.session_state.model.parameters())
            adaptation_params = len(st.session_state.learner.adaptation_params) if st.session_state.learner else 0
            
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Adaptation Parameters", f"{adaptation_params:,}")
            st.metric("Parameter Efficiency", f"{adaptation_params/total_params:.4f}")
        
        # Current metrics
        st.subheader("Current Metrics")
        if st.session_state.metrics.predictions:
            accuracy_metrics = st.session_state.metrics.get_accuracy_metrics()
            performance_metrics = st.session_state.metrics.get_performance_metrics()
            
            st.metric("Accuracy", f"{accuracy_metrics['accuracy']:.4f}")
            st.metric("Mean Latency", f"{performance_metrics['latency_mean']:.2f} ms")
            st.metric("Throughput", f"{performance_metrics['throughput_fps']:.2f} FPS")
            st.metric("Memory Usage", f"{performance_metrics['memory_mean']:.2f} MB")
        
        # Learning stats
        st.subheader("Learning Statistics")
        if st.session_state.learner:
            stats = st.session_state.learner.get_adaptation_stats()
            st.metric("Total Samples", stats['total_samples'])
            st.metric("Buffer Size", stats['buffer_size'])
            st.metric("Update Frequency", update_frequency)
    
    # Streaming simulation
    if st.session_state.is_streaming and st.session_state.streamer:
        # Simulate streaming data
        if sensor_type == "Camera":
            # Generate synthetic image data
            data = torch.randn(1, 1, 28, 28)
            target = np.random.randint(0, 10)
        else:  # IMU
            # Generate synthetic IMU data
            data = torch.randn(1, 6)
            target = np.random.randint(0, 10)
        
        # Run inference
        prediction, latency = simulate_inference(st.session_state.model, data)
        
        # Add to metrics
        st.session_state.metrics.add_prediction(
            prediction=prediction,
            target=target,
            latency=latency,
            memory_usage=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0,
        )
        
        # Add to learner
        st.session_state.learner.add_sample(data, torch.tensor([target]))
        
        # Update learning curve
        if st.session_state.metrics.predictions:
            recent_accuracy = st.session_state.metrics.get_accuracy_metrics()['accuracy']
            st.session_state.curve_tracker.add_measurement(
                accuracy=recent_accuracy,
                loss=0.0,  # Simplified for demo
                latency=latency,
                sample_count=st.session_state.learner.sample_count,
            )
        
        # Auto-refresh for real-time updates
        time.sleep(1.0 / sampling_rate)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>On-Device Learning Implementation | Edge AI & IoT Project</p>
    <p>This demo showcases real-time adaptation capabilities for edge devices</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
