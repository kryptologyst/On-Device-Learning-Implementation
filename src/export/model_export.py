"""Model export and compilation for edge deployment."""

import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class ModelExporter:
    """Export PyTorch models to various edge runtime formats."""
    
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """Initialize model exporter.
        
        Args:
            model: PyTorch model to export.
            device: Device for computation.
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def export_to_onnx(
        self,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 11,
        optimize: bool = True,
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            input_shape: Input tensor shape.
            output_path: Output file path.
            opset_version: ONNX opset version.
            optimize: Whether to optimize the model.
            
        Returns:
            Path to exported ONNX model.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Optimize if requested
        if optimize:
            self._optimize_onnx(output_path)
        
        return output_path
    
    def _optimize_onnx(self, model_path: str) -> None:
        """Optimize ONNX model for edge deployment.
        
        Args:
            model_path: Path to ONNX model.
        """
        if not ONNX_AVAILABLE:
            return
        
        # Load model
        model = onnx.load(model_path)
        
        # Basic optimizations
        from onnx import optimizer
        passes = ['eliminate_identity', 'eliminate_nop_transpose', 'fuse_consecutive_transposes']
        
        try:
            optimized_model = optimizer.optimize(model, passes)
            onnx.save(optimized_model, model_path)
        except Exception as e:
            print(f"Warning: ONNX optimization failed: {e}")
    
    def export_to_tflite(
        self,
        input_shape: Tuple[int, ...],
        output_path: str,
        quantize: bool = True,
        quantize_mode: str = "int8",
    ) -> str:
        """Export model to TensorFlow Lite format.
        
        Args:
            input_shape: Input tensor shape.
            output_path: Output file path.
            quantize: Whether to quantize the model.
            quantize_mode: Quantization mode ('int8', 'float16').
            
        Returns:
            Path to exported TFLite model.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        # First export to ONNX, then convert to TFLite
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            onnx_path = tmp_file.name
        
        try:
            # Export to ONNX first
            self.export_to_onnx(input_shape, onnx_path, optimize=False)
            
            # Convert ONNX to TensorFlow
            tf_model = self._onnx_to_tf(onnx_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
            
            if quantize:
                if quantize_mode == "int8":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.int8]
                elif quantize_mode == "float16":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            return output_path
            
        finally:
            # Cleanup temporary files
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
    
    def _onnx_to_tf(self, onnx_path: str) -> str:
        """Convert ONNX model to TensorFlow format.
        
        Args:
            onnx_path: Path to ONNX model.
            
        Returns:
            Path to TensorFlow model.
        """
        try:
            import tf2onnx
            from tf2onnx import convert
            
            # Convert ONNX to TensorFlow
            tf_path = onnx_path.replace('.onnx', '_tf')
            convert.from_onnx(onnx_path, tf_path)
            
            return tf_path
        except ImportError:
            raise ImportError("tf2onnx not available. Install with: pip install tf2onnx")
    
    def export_to_coreml(
        self,
        input_shape: Tuple[int, ...],
        output_path: str,
        quantize: bool = True,
    ) -> str:
        """Export model to CoreML format for iOS deployment.
        
        Args:
            input_shape: Input tensor shape.
            output_path: Output file path.
            quantize: Whether to quantize the model.
            
        Returns:
            Path to exported CoreML model.
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML Tools not available. Install with: pip install coremltools")
        
        # First export to ONNX
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            onnx_path = tmp_file.name
        
        try:
            self.export_to_onnx(input_shape, onnx_path, optimize=False)
            
            # Convert ONNX to CoreML
            model = ct.convert(onnx_path)
            
            # Quantize if requested
            if quantize:
                model = ct.models.neural_network.quantization_utils.quantize_weights(
                    model, nbits=8
                )
            
            # Save CoreML model
            model.save(output_path)
            
            return output_path
            
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
    
    def benchmark_exported_model(
        self,
        model_path: str,
        format_type: str,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark exported model performance.
        
        Args:
            model_path: Path to exported model.
            format_type: Model format ('onnx', 'tflite', 'coreml').
            input_shape: Input tensor shape.
            num_runs: Number of benchmark runs.
            
        Returns:
            Dictionary with benchmark results.
        """
        latencies = []
        
        if format_type == "onnx":
            latencies = self._benchmark_onnx(model_path, input_shape, num_runs)
        elif format_type == "tflite":
            latencies = self._benchmark_tflite(model_path, input_shape, num_runs)
        elif format_type == "coreml":
            latencies = self._benchmark_coreml(model_path, input_shape, num_runs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return {
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "throughput_fps": 1000.0 / np.mean(latencies),
        }
    
    def _benchmark_onnx(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
    ) -> List[float]:
        """Benchmark ONNX model."""
        if not ONNX_AVAILABLE:
            return []
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(model_path)
        
        # Create dummy input
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {'input': dummy_input})
        
        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, {'input': dummy_input})
            latencies.append((time.time() - start_time) * 1000)
        
        return latencies
    
    def _benchmark_tflite(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
    ) -> List[float]:
        """Benchmark TFLite model."""
        if not TENSORFLOW_AVAILABLE:
            return []
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create dummy input
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            latencies.append((time.time() - start_time) * 1000)
        
        return latencies
    
    def _benchmark_coreml(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int,
    ) -> List[float]:
        """Benchmark CoreML model."""
        if not COREML_AVAILABLE:
            return []
        
        # Load CoreML model
        model = ct.models.MLModel(model_path)
        
        # Create dummy input
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            model.predict({'input': dummy_input})
        
        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            model.predict({'input': dummy_input})
            latencies.append((time.time() - start_time) * 1000)
        
        return latencies


class EdgeDeploymentManager:
    """Manage deployment to various edge devices."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize deployment manager.
        
        Args:
            config: Deployment configuration.
        """
        self.config = config
        self.device_configs = config.device
        
    def get_deployment_config(self, device_name: str) -> DictConfig:
        """Get deployment configuration for specific device.
        
        Args:
            device_name: Name of the target device.
            
        Returns:
            Device-specific configuration.
        """
        if device_name not in self.device_configs:
            raise ValueError(f"Unknown device: {device_name}")
        
        return self.device_configs[device_name]
    
    def generate_deployment_script(
        self,
        device_name: str,
        model_path: str,
        output_dir: str,
    ) -> str:
        """Generate deployment script for target device.
        
        Args:
            device_name: Target device name.
            model_path: Path to exported model.
            output_dir: Output directory for deployment files.
            
        Returns:
            Path to generated deployment script.
        """
        device_config = self.get_deployment_config(device_name)
        
        if device_name == "raspberry_pi":
            return self._generate_raspberry_pi_script(device_config, model_path, output_dir)
        elif device_name == "jetson_nano":
            return self._generate_jetson_script(device_config, model_path, output_dir)
        elif device_name == "android":
            return self._generate_android_script(device_config, model_path, output_dir)
        elif device_name == "ios":
            return self._generate_ios_script(device_config, model_path, output_dir)
        else:
            raise ValueError(f"Unsupported device: {device_name}")
    
    def _generate_raspberry_pi_script(
        self,
        device_config: DictConfig,
        model_path: str,
        output_dir: str,
    ) -> str:
        """Generate Raspberry Pi deployment script."""
        script_content = f"""#!/bin/bash
# Raspberry Pi Deployment Script
# Generated for {device_config.os}

echo "Deploying model to Raspberry Pi..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install onnxruntime numpy opencv-python

# Copy model
cp {model_path} ./model.onnx

# Create inference script
cat > inference.py << 'EOF'
import onnxruntime as ort
import numpy as np
import cv2
import time

class EdgeInference:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def preprocess(self, image):
        # Resize and normalize image
        image = cv2.resize(image, (28, 28))
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)
    
    def predict(self, image):
        input_data = self.preprocess(image)
        start_time = time.time()
        outputs = self.session.run(None, {{self.input_name: input_data}})
        latency = (time.time() - start_time) * 1000
        return outputs[0], latency

# Example usage
if __name__ == "__main__":
    inference = EdgeInference("model.onnx")
    
    # Simulate camera input
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run inference
        prediction, latency = inference.predict(gray)
        
        print(f"Prediction: {{np.argmax(prediction)}}, Latency: {{latency:.2f}}ms")
        
        # Display result
        cv2.putText(frame, f"Class: {{np.argmax(prediction)}}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {{latency:.1f}}ms", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Edge Inference", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
EOF

chmod +x inference.py
echo "Deployment complete! Run: python3 inference.py"
"""
        
        script_path = os.path.join(output_dir, f"deploy_{device_name}.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def _generate_jetson_script(
        self,
        device_config: DictConfig,
        model_path: str,
        output_dir: str,
    ) -> str:
        """Generate Jetson deployment script."""
        script_content = f"""#!/bin/bash
# NVIDIA Jetson Deployment Script
# Generated for {device_config.os}

echo "Deploying model to Jetson Nano..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Install TensorRT (if available)
if command -v tensorrt &> /dev/null; then
    echo "TensorRT found, optimizing model..."
    # Add TensorRT optimization commands here
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install onnxruntime-gpu numpy opencv-python

# Copy model
cp {model_path} ./model.onnx

# Create optimized inference script
cat > inference.py << 'EOF'
import onnxruntime as ort
import numpy as np
import cv2
import time

class JetsonInference:
    def __init__(self, model_path):
        # Use GPU provider if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
    def preprocess(self, image):
        image = cv2.resize(image, (28, 28))
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)
    
    def predict(self, image):
        input_data = self.preprocess(image)
        start_time = time.time()
        outputs = self.session.run(None, {{self.input_name: input_data}})
        latency = (time.time() - start_time) * 1000
        return outputs[0], latency

# Example usage
if __name__ == "__main__":
    inference = JetsonInference("model.onnx")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prediction, latency = inference.predict(gray)
        
        print(f"Prediction: {{np.argmax(prediction)}}, Latency: {{latency:.2f}}ms")
        
        cv2.putText(frame, f"Class: {{np.argmax(prediction)}}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {{latency:.1f}}ms", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Jetson Inference", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
EOF

chmod +x inference.py
echo "Deployment complete! Run: python3 inference.py"
"""
        
        script_path = os.path.join(output_dir, f"deploy_{device_name}.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def _generate_android_script(
        self,
        device_config: DictConfig,
        model_path: str,
        output_dir: str,
    ) -> str:
        """Generate Android deployment script."""
        script_content = f"""#!/bin/bash
# Android Deployment Script
# Generated for {device_config.os}

echo "Preparing Android deployment..."

# Create Android project structure
mkdir -p android_app/app/src/main/assets
mkdir -p android_app/app/src/main/java/com/edgeai/inference

# Copy TFLite model
cp {model_path} android_app/app/src/main/assets/model.tflite

# Create Android inference class
cat > android_app/app/src/main/java/com/edgeai/inference/EdgeInference.java << 'EOF'
package com.edgeai.inference;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import org.tensorflow.lite.Interpreter;

public class EdgeInference {{
    private Interpreter tflite;
    private ByteBuffer inputBuffer;
    private float[][] outputArray;
    
    public EdgeInference(Context context) throws IOException {{
        tflite = new Interpreter(loadModelFile(context));
        inputBuffer = ByteBuffer.allocateDirect(4 * 1 * 28 * 28);
        inputBuffer.order(ByteOrder.nativeOrder());
        outputArray = new float[1][10];
    }}
    
    private MappedByteBuffer loadModelFile(Context context) throws IOException {{
        FileInputStream inputStream = new FileInputStream(context.getAssets().open("model.tflite"));
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = 0;
        long declaredLength = fileChannel.size();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }}
    
    public float[] predict(Bitmap bitmap) {{
        // Preprocess image
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
        convertBitmapToByteBuffer(resized);
        
        // Run inference
        tflite.run(inputBuffer, outputArray);
        
        return outputArray[0];
    }}
    
    private void convertBitmapToByteBuffer(Bitmap bitmap) {{
        inputBuffer.rewind();
        int[] pixels = new int[28 * 28];
        bitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);
        
        for (int pixel : pixels) {{
            float normalizedPixel = (pixel & 0xFF) / 255.0f;
            inputBuffer.putFloat(normalizedPixel);
        }}
    }}
    
    public void close() {{
        tflite.close();
    }}
}}
EOF

# Create build.gradle
cat > android_app/app/build.gradle << 'EOF'
android {{
    compileSdkVersion 33
    
    defaultConfig {{
        applicationId "com.edgeai.inference"
        minSdkVersion 21
        targetSdkVersion 33
        versionCode 1
        versionName "1.0"
    }}
    
    dependencies {{
        implementation 'org.tensorflow:tensorflow-lite:2.10.0'
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.10.0'
    }}
}}
EOF

echo "Android project created in android_app/"
echo "Import into Android Studio to build and deploy"
"""
        
        script_path = os.path.join(output_dir, f"deploy_{device_name}.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def _generate_ios_script(
        self,
        device_config: DictConfig,
        model_path: str,
        output_dir: str,
    ) -> str:
        """Generate iOS deployment script."""
        script_content = f"""#!/bin/bash
# iOS Deployment Script
# Generated for {device_config.os}

echo "Preparing iOS deployment..."

# Create iOS project structure
mkdir -p ios_app/EdgeAI/Models
mkdir -p ios_app/EdgeAI/Inference

# Copy CoreML model
cp {model_path} ios_app/EdgeAI/Models/model.mlmodel

# Create iOS inference class
cat > ios_app/EdgeAI/Inference/EdgeInference.swift << 'EOF'
import UIKit
import CoreML
import Vision

class EdgeInference {{
    private let model: MLModel
    private let visionModel: VNCoreMLModel
    
    init() throws {{
        guard let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodel") else {{
            throw InferenceError.modelNotFound
        }}
        
        model = try MLModel(contentsOf: modelURL)
        visionModel = try VNCoreMLModel(for: model)
    }}
    
    func predict(image: UIImage, completion: @escaping (Result<[Float], Error>) -> Void) {{
        guard let cgImage = image.cgImage else {{
            completion(.failure(InferenceError.invalidImage))
            return
        }}
        
        let request = VNCoreMLRequest(model: visionModel) {{ request, error in
            if let error = error {{
                completion(.failure(error))
                return
            }}
            
            guard let results = request.results as? [VNClassificationObservation] else {{
                completion(.failure(InferenceError.invalidResults))
                return
            }}
            
            let predictions = results.map {{ $0.confidence }}
            completion(.success(predictions))
        }}
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }}
}}

enum InferenceError: Error {{
    case modelNotFound
    case invalidImage
    case invalidResults
}}
EOF

# Create Info.plist
cat > ios_app/EdgeAI/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Edge AI</string>
    <key>CFBundleIdentifier</key>
    <string>com.edgeai.inference</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSRequiresIPhoneOS</key>
    <true/>
    <key>UILaunchStoryboardName</key>
    <string>LaunchScreen</string>
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
    </array>
</dict>
</plist>
EOF

echo "iOS project created in ios_app/"
echo "Import into Xcode to build and deploy"
"""
        
        script_path = os.path.join(output_dir, f"deploy_{device_name}.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path


# Import time for benchmarking
import time
