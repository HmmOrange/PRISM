# Model Abstractions System

Hệ thống Model Abstractions cung cấp lớp trừu tượng thống nhất để quản lý và sử dụng các model từ nhiều framework khác nhau (HuggingFace, ONNX, PyTorch, TensorFlow, YOLO, etc.).

## 🎯 Mục tiêu

- **Unified Interface**: Tất cả models dùng chung 1 interface bất kể framework
- **Framework Agnostic**: Hỗ trợ nhiều framework mà không cần thay đổi code logic
- **Performance Optimization**: Auto convert sang ONNX để tăng tốc inference
- **Smart Memory Management**: Quản lý RAM và disk space tự động
- **Easy Extensibility**: Dễ dàng thêm support cho framework mới

## 🏗️ Architecture

### Core Components

1. **BaseModelWrapper**: Lớp trừu tượng cơ bản cho tất cả model wrappers
2. **ModelRegistry**: Registry pattern để quản lý các wrapper classes
3. **ModelManager**: Quản lý lifecycle của models (load/unload/caching)
4. **Framework-specific Wrappers**: Implement cho từng framework cụ thể
5. **ModelConverter**: Convert models giữa các format khác nhau

```
┌─────────────────────────────────────────────────────────┐
│                    ModelManager                        │
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  Memory Mgmt    │  │     Model Registry          │   │
│  │  - RAM limits   │  │  ┌─────────────────────────┐ │   │
│  │  - Disk limits  │  │  │    HuggingFace         │ │   │
│  │  - Auto cleanup │  │  │    ONNX                │ │   │
│  └─────────────────┘  │  │    PyTorch             │ │   │
│                       │  │    TensorFlow          │ │   │
│  ┌─────────────────┐  │  │    YOLO                │ │   │
│  │  Auto Convert   │  │  │    Custom Pipelines    │ │   │
│  │  - To ONNX      │  │  └─────────────────────────┘ │   │
│  │  - Framework    │  └─────────────────────────────┘   │
│  │    Detection    │                                    │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                BaseModelWrapper                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Common Interface:                                  │ │
│  │  - load_model()                                     │ │
│  │  - unload_model()                                   │ │
│  │  - predict(inputs)                                  │ │
│  │  - to_device(device)                                │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Usage Examples

### Basic Usage

```python
from server.model_abstractions import ModelManager, ModelType, ModelFramework

# Initialize Model Manager
manager = ModelManager(
    max_models_in_ram=5,
    max_models_in_disk=10,
    auto_convert_to_onnx=True  # Auto convert để optimize performance
)

# Register models
manager.register_model(
    model_id="bert-base-uncased",
    model_type=ModelType.TEXT_CLASSIFICATION,
    framework=ModelFramework.HUGGINGFACE
)

# Use unified interface cho bất kỳ model nào
result = manager.predict(
    model_id="bert-base-uncased",
    inputs={"text": "This is a great product!"},
    device="cuda:0"
)
```

### Working with Different Frameworks

```python
# HuggingFace Model
hf_result = manager.predict("bert-classifier", {"text": "Hello world"})

# ONNX Model (same interface!)
onnx_result = manager.predict("resnet50-onnx", {"image": image_data})

# YOLO Model (same interface!)
yolo_result = manager.predict("yolo-detector", {"image": image_data})

# Custom Tabular Model (same interface!)
tabular_result = manager.predict("fraud-detector", {"row": feature_dict})
```

### Auto Conversion to ONNX

```python
# Enable auto conversion
manager = ModelManager(auto_convert_to_onnx=True)

# Model sẽ được tự động convert sang ONNX khi load
# -> Faster inference, smaller memory footprint
result = manager.predict("any-model-id", inputs)
```

## 📦 Framework Support

### Supported Frameworks

| Framework | Models | Status | Performance | Notes |
|-----------|--------|--------|-------------|-------|
| **HuggingFace** | Text, Image, Audio, Video | ✅ | Good | Native support |
| **ONNX** | All types | ✅ | Excellent | Optimized runtime |
| **PyTorch** | Custom models | ✅ | Good | Via conversion |
| **TensorFlow** | All types | ✅ | Good | Via tf2onnx |
| **YOLO** | Object Detection | ✅ | Excellent | Ultralytics |
| **scikit-learn** | Tabular | ✅ | Good | Via skl2onnx |
| **XGBoost** | Tabular | 🚧 | Good | Planned |
| **LightGBM** | Tabular | 🚧 | Good | Planned |

### Model Types Supported

- **Text**: Classification, NER, Translation, Summarization, QA, Generation
- **Image**: Classification, Object Detection, Image-to-Text
- **Audio**: Classification, Speech Recognition
- **Video**: Classification
- **Tabular**: Classification, Regression

## 🔧 Adding New Framework Support

### Step 1: Create Wrapper Class

```python
from .base_model import BaseModelWrapper, ModelFramework, ModelType

class MyFrameworkWrapper(BaseModelWrapper):
    def __init__(self, model_path, model_type, device="cpu", **kwargs):
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            framework=ModelFramework.MY_FRAMEWORK,
            device=device
        )
    
    def load_model(self):
        # Load model implementation
        pass
    
    def unload_model(self):
        # Unload model implementation
        pass
    
    def predict(self, inputs):
        # Prediction implementation
        pass
    
    def to_device(self, device):
        # Device transfer implementation
        pass
```

### Step 2: Register Wrapper

```python
from .base_model import ModelRegistry

# Register cho specific model types
ModelRegistry.register(
    ModelFramework.MY_FRAMEWORK, 
    ModelType.TEXT_CLASSIFICATION, 
    MyFrameworkWrapper
)
```

### Step 3: Add Converter (Optional)

```python
from .model_converter import ModelConverter

def convert_my_framework_to_onnx(model_path, output_path, model_type):
    # Conversion logic
    pass

# Add to AutoConverter
```

## ⚡ Performance Optimizations

### ONNX Conversion Benefits

- **Faster Inference**: 2-5x speedup compared to original frameworks
- **Smaller Memory**: Optimized model representation
- **Cross-platform**: Run anywhere with ONNX Runtime
- **Hardware Acceleration**: GPU, NPU, specialized chips

### Memory Management

```python
# Configure memory limits
manager = ModelManager(
    max_models_in_ram=5,      # Limit loaded models
    max_models_in_disk=20,    # Limit cached models
    auto_convert_to_onnx=True # Use optimized format
)

# Models tự động load/unload theo LRU policy
```

### Caching Strategy

1. **RAM Cache**: Keep frequently used models in memory
2. **Disk Cache**: Store downloaded models locally
3. **LRU Eviction**: Remove oldest unused models when limits reached
4. **Smart Cleanup**: Prioritize models not in RAM for disk cleanup

## 🛠️ Configuration

### Environment Variables

```bash
# Model server settings
export MAX_MODELS_IN_RAM=5
export MAX_MODELS_IN_DISK=10
export MODELS_DIRECTORY="/path/to/models"
export AUTO_CONVERT_TO_ONNX=true
export DEFAULT_DEVICE="cuda:0"
```

### Manager Configuration

```python
manager = ModelManager(
    max_models_in_ram=5,
    max_models_in_disk=10,
    models_directory="./models",
    auto_convert_to_onnx=True
)
```

## 📊 Monitoring & Logging

### Model Status

```python
# List all models
models = manager.list_models()
for model in models:
    print(f"{model['model_id']}: {model['is_loaded']}")

# Get detailed info
info = manager.get_model_info("model-id")
print(f"Framework: {info['framework']}")
print(f"Device: {info['device']}")
print(f"Last used: {info['lasted_used']}")
```

### Performance Metrics

- Model load/unload times
- Inference latency
- Memory usage
- Cache hit rates
- Conversion success rates

## 🔄 Migration from Old System

### Step-by-Step Migration

1. **Install Dependencies**:
   ```bash
   pip install onnxruntime transformers torch tensorflow
   ```

2. **Update Imports**:
   ```python
   # Old
   from server.api_router import get_pipe
   
   # New
   from server.model_abstractions import ModelManager
   ```

3. **Replace Logic**:
   ```python
   # Old
   pipe = get_pipe(model_id)
   result = pipe(inputs)
   
   # New
   result = manager.predict(model_id, inputs, device)
   ```

4. **Update Configuration**:
   - Move model configs to new format
   - Set memory limits
   - Enable ONNX conversion

### Backward Compatibility

- API endpoints remain the same
- Response format unchanged
- Gradual migration possible

## 🔍 Troubleshooting

### Common Issues

1. **Model Not Found**: Check if model is registered
2. **Memory Issues**: Reduce `max_models_in_ram`
3. **Conversion Fails**: Check framework compatibility
4. **Slow Loading**: Enable ONNX conversion
5. **Device Errors**: Verify CUDA availability

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logs for debugging
manager = ModelManager(debug=True)
```

## 🚧 Future Roadmap

### Planned Features

- [ ] **Multi-GPU Support**: Distribute models across GPUs
- [ ] **Model Quantization**: INT8/FP16 optimization
- [ ] **Batch Inference**: Process multiple requests together
- [ ] **Model Versioning**: Support multiple versions
- [ ] **A/B Testing**: Compare model performance
- [ ] **Distributed Inference**: Across multiple nodes
- [ ] **Model Monitoring**: Performance tracking
- [ ] **Auto Scaling**: Based on load

### Framework Expansion

- [ ] **XGBoost/LightGBM**: Gradient boosting models
- [ ] **MLflow**: Model registry integration
- [ ] **TensorRT**: NVIDIA optimization
- [ ] **CoreML**: Apple devices
- [ ] **TensorFlow Lite**: Mobile deployment

## 📝 Contributing

1. Fork the repository
2. Create feature branch
3. Add new framework wrapper
4. Write tests
5. Update documentation
6. Submit pull request

### Guidelines

- Follow existing patterns
- Add comprehensive tests
- Document new features
- Maintain backward compatibility

---

**Tóm lại**: Hệ thống Model Abstractions mới cung cấp interface thống nhất, tối ưu hóa performance thông qua ONNX, và quản lý memory thông minh - giúp scale từ development đến production một cách dễ dàng. 