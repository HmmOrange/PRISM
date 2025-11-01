#!/usr/bin/env python3
"""
Test script để demo hệ thống Model Abstractions mới
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.model_abstractions import (
    ModelManager, ModelFramework, ModelType, ModelRegistry
)


def test_model_registry():
    """Test Model Registry functionality"""
    print("=== Testing Model Registry ===")
    
    # List supported combinations
    combinations = ModelRegistry.list_supported_combinations()
    print(f"Supported combinations: {len(combinations)}")
    
    for combo in combinations[:5]:  # Show first 5
        print(f"  - {combo['framework']} + {combo['model_type']}")
    
    print()


def test_model_manager():
    """Test Model Manager functionality"""
    print("=== Testing Model Manager ===")
    
    # Initialize Model Manager
    manager = ModelManager(
        max_models_in_ram=2,
        max_models_in_disk=5,
        models_directory="./test_models",
        auto_convert_to_onnx=False
    )
    
    # Register một số test models
    print("Registering test models...")
    
    # Register HuggingFace text model
    success = manager.register_model(
        model_id="test-text-classifier",
        model_type=ModelType.TEXT_CLASSIFICATION,
        framework=ModelFramework.HUGGINGFACE,
        metadata={"description": "Test text classification model"}
    )
    print(f"HuggingFace text model registered: {success}")
    
    # Register ONNX image model
    success = manager.register_model(
        model_id="test-image-classifier",
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=ModelFramework.ONNX,
        metadata={"description": "Test ONNX image classification model"}
    )
    print(f"ONNX image model registered: {success}")
    
    # List all models
    models = manager.list_models()
    print(f"\nRegistered models: {len(models)}")
    for model in models:
        print(f"  - {model['model_id']} ({model['framework']}/{model['model_type']})")
    
    print()


def test_conversion_workflow():
    """Test model conversion workflow"""
    print("=== Testing Model Conversion Workflow ===")
    
    from server.model_abstractions.model_converter import AutoConverter, ModelConverter
    
    # Test framework detection
    test_paths = [
        "./fake_hf_model",
        "./fake_onnx_model", 
        "./fake_pytorch_model"
    ]
    
    for path in test_paths:
        framework = AutoConverter.detect_model_framework(path)
        print(f"Framework detection for {path}: {framework}")
    
    print()


def demo_unified_interface():
    """Demo unified interface cho các framework khác nhau"""
    print("=== Demo Unified Interface ===")
    
    manager = ModelManager(models_directory="./demo_models")
    
    # Demo register models từ các framework khác nhau
    models_to_register = [
        {
            "model_id": "bert-base-uncased",
            "model_type": ModelType.TEXT_CLASSIFICATION,
            "framework": ModelFramework.HUGGINGFACE,
            "description": "BERT text classifier"
        },
        {
            "model_id": "resnet50-onnx",
            "model_type": ModelType.IMAGE_CLASSIFICATION,
            "framework": ModelFramework.ONNX,
            "description": "ResNet50 ONNX model"
        },
        {
            "model_id": "yolo-detector",
            "model_type": ModelType.OBJECT_DETECTION,
            "framework": ModelFramework.YOLO,
            "description": "YOLO object detector"
        },
        {
            "model_id": "tabular-clf",
            "model_type": ModelType.TABULAR_CLASSIFICATION,
            "framework": ModelFramework.TENSORFLOW,
            "description": "Tabular classifier"
        }
    ]
    
    print("Registering diverse models...")
    for model_config in models_to_register:
        success = manager.register_model(
            model_id=model_config["model_id"],
            model_type=model_config["model_type"],
            framework=model_config["framework"],
            metadata={"description": model_config["description"]}
        )
        print(f"  ✓ {model_config['model_id']} ({model_config['framework'].value}): {success}")
    
    # Show unified interface
    print(f"\nUnified interface managing {len(manager.list_models())} models:")
    for model in manager.list_models():
        print(f"  - {model['model_id']}")
        print(f"    Framework: {model['framework']}")
        print(f"    Type: {model['model_type']}")
        print(f"    Loaded: {model['is_loaded']}")
        print()
    
    print("All models can be used with the same interface:")
    print("  manager.predict(model_id, inputs, device)")
    print("  manager.load_model(model_id, device)")
    print("  manager.unload_model(model_id)")
    
    print()


def show_benefits():
    """Show benefits của hệ thống mới"""
    print("=== Benefits của Model Abstractions System ===")
    
    benefits = [
        "🎯 Unified Interface: Tất cả models dùng chung 1 interface",
        "🔧 Framework Agnostic: Hỗ trợ HuggingFace, ONNX, PyTorch, TensorFlow, YOLO, etc.",
        "⚡ ONNX Optimization: Auto convert sang ONNX để tăng tốc inference",
        "🧠 Smart Memory Management: Tự động quản lý RAM và disk space",
        "📦 Plugin Architecture: Dễ dàng thêm support cho framework mới",
        "🔄 Hot Swapping: Load/unload models on-demand",
        "📊 Model Registry: Centralized management của tất cả models",
        "🛠️ Auto Detection: Tự động detect framework từ model files",
        "🎛️ Configurable: Flexible configuration cho different use cases",
        "📈 Scalable: Có thể scale từ development đến production"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print()


def show_usage_examples():
    """Show concrete usage examples"""
    print("=== Usage Examples ===")
    
    examples = [
        "# Sử dụng HuggingFace model",
        "result = manager.predict('bert-base-uncased', {'text': 'Hello world'})",
        "",
        "# Sử dụng ONNX model với cùng interface",
        "result = manager.predict('resnet50-onnx', {'image': image_data})",
        "",
        "# Automatic conversion sang ONNX",
        "manager = ModelManager(auto_convert_to_onnx=True)",
        "",
        "# Custom preprocessing cho specific models",
        "model.metadata['preprocessing'] = {'normalize': True, 'resize': (224, 224)}",
        "",
        "# Batch inference với multiple models",
        "for model_id in ['model1', 'model2', 'model3']:",
        "    result = manager.predict(model_id, inputs)"
    ]
    
    for example in examples:
        print(f"    {example}")
    
    print()


def main():
    """Main test function"""
    print("🚀 Testing New Model Abstractions System\n")
    
    test_model_registry()
    test_model_manager()
    test_conversion_workflow()
    demo_unified_interface()
    show_benefits()
    show_usage_examples()
    
    print("✅ Test completed! The new system provides:")
    print("   - Unified interface cho tất cả frameworks")
    print("   - ONNX conversion cho performance optimization")
    print("   - Smart memory management")
    print("   - Easy extensibility cho new frameworks")
    print("   - Production-ready scalability")


if __name__ == "__main__":
    main()