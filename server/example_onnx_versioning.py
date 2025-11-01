#!/usr/bin/env python3
"""
Example script để demo ONNX Version Management System

Demonstrates các deployment strategies khác nhau cho models với 
ONNX version requirements khác nhau.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from server.model_abstractions.onnx_version_manager import (
    ONNXVersionManager, ONNXVersionSpec, VersionedONNXWrapper
)
from server.model_abstractions.deployment_configs import (
    DeploymentConfigManager, ModelDeploymentConfig, DeploymentStrategy,
    recommend_deployment_strategy
)
from server.model_abstractions.base_model import ModelType


def demo_version_management():
    """Demo basic version management"""
    print("🔧 ONNX Version Management Demo")
    print("=" * 50)
    
    # Initialize Version Manager
    version_manager = ONNXVersionManager("./demo_onnx_environments")
    
    # Register model version requirements
    models_with_versions = [
        {
            "model_id": "bert-base-uncased",
            "spec": ONNXVersionSpec(
                onnx_version="1.15.0",
                opset_version=17,
                providers=["CPUExecutionProvider"],
                python_version="3.9"
            )
        },
        {
            "model_id": "resnet50-legacy",
            "spec": ONNXVersionSpec(
                onnx_version="1.8.0", 
                opset_version=11,
                providers=["CPUExecutionProvider"],
                python_version="3.7",
                gpu_support=False
            )
        },
        {
            "model_id": "yolo-v8-gpu",
            "spec": ONNXVersionSpec(
                onnx_version="1.12.0",
                opset_version=13,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                python_version="3.8",
                gpu_support=True
            )
        }
    ]
    
    # Register models
    for model_info in models_with_versions:
        version_manager.register_model_version_requirement(
            model_info["model_id"],
            model_info["spec"]
        )
        print(f"✅ Registered {model_info['model_id']} -> ONNX {model_info['spec'].onnx_version}")
    
    # Show version mappings
    print("\n📋 Model Version Mappings:")
    for model_id in ["bert-base-uncased", "resnet50-legacy", "yolo-v8-gpu"]:
        spec = version_manager.get_version_for_model(model_id)
        if spec:
            print(f"  {model_id}:")
            print(f"    ONNX: {spec.onnx_version}")
            print(f"    Opset: {spec.opset_version}")
            print(f"    Python: {spec.python_version}")
            print(f"    GPU: {spec.gpu_support}")
    
    print()


def demo_deployment_strategies():
    """Demo deployment strategy configurations"""
    print("🚀 Deployment Strategies Demo")
    print("=" * 50)
    
    # Initialize Deployment Config Manager
    config_manager = DeploymentConfigManager("./demo_deployment_configs.yaml")
    
    # Example scenarios và recommended strategies
    scenarios = [
        {
            "name": "Development - Simple Text Model",
            "params": {
                "model_complexity": "simple",
                "expected_load": "low", 
                "isolation_required": False,
                "development_mode": True
            }
        },
        {
            "name": "Production - GPU Image Model",
            "params": {
                "model_complexity": "complex",
                "expected_load": "medium",
                "gpu_required": True,
                "development_mode": False
            }
        },
        {
            "name": "High Load - Scalable Service",
            "params": {
                "model_complexity": "medium",
                "expected_load": "high",
                "isolation_required": True,
                "development_mode": False
            }
        },
        {
            "name": "Legacy Model - Isolated Environment",
            "params": {
                "model_complexity": "medium",
                "expected_load": "low",
                "isolation_required": True,
                "development_mode": True
            }
        }
    ]
    
    # Show recommendations
    for scenario in scenarios:
        strategy = recommend_deployment_strategy(**scenario["params"])
        print(f"📊 {scenario['name']}")
        print(f"   Recommended: {strategy.value}")
        print(f"   Params: {scenario['params']}")
        print()
    
    # Create example configurations
    example_configs = [
        ModelDeploymentConfig(
            model_id="bert-text-classifier",
            onnx_version="1.15.0",
            opset_version=17,
            strategy=DeploymentStrategy.CONDA,
            memory_limit="4Gi",
            cpu_limit="2000m"
        ),
        ModelDeploymentConfig(
            model_id="resnet50-image-classifier",
            onnx_version="1.12.0", 
            opset_version=13,
            strategy=DeploymentStrategy.DOCKER,
            memory_limit="6Gi",
            gpu_required=True,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        ),
        ModelDeploymentConfig(
            model_id="high-traffic-api",
            onnx_version="1.15.0",
            opset_version=17,
            strategy=DeploymentStrategy.KUBERNETES,
            replicas=3,
            auto_scaling=True,
            min_replicas=2,
            max_replicas=10
        )
    ]
    
    # Register configurations
    for config in example_configs:
        config_manager.register_model_config(config)
        print(f"✅ Registered deployment config for {config.model_id}")
    
    print()


def demo_deployment_file_generation():
    """Demo deployment file generation"""
    print("📁 Deployment File Generation Demo")
    print("=" * 50)
    
    config_manager = DeploymentConfigManager("./demo_deployment_configs.yaml")
    
    # Generate deployment files cho different strategies
    models_to_deploy = ["bert-text-classifier", "resnet50-image-classifier", "high-traffic-api"]
    
    for model_id in models_to_deploy:
        config = config_manager.get_config(model_id)
        if config:
            output_dir = f"./deployments/{model_id}"
            
            print(f"🔨 Generating files for {model_id} ({config.strategy.value})")
            
            try:
                config_manager.generate_deployment_files(model_id, output_dir)
                
                # List generated files
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    print(f"   Generated: {', '.join(files)}")
                else:
                    print("   No files generated")
            except Exception as e:
                print(f"   Error: {e}")
        
        print()


def demo_conda_environment_setup():
    """Demo Conda environment setup"""
    print("🐍 Conda Environment Setup Demo")
    print("=" * 50)
    
    version_manager = ONNXVersionManager("./demo_onnx_environments")
    
    # Create conda environments cho different ONNX versions
    version_specs = {
        "onnx_1_15": ONNXVersionSpec(
            onnx_version="1.15.0",
            opset_version=17,
            providers=["CPUExecutionProvider"],
            python_version="3.9"
        ),
        "onnx_1_8_legacy": ONNXVersionSpec(
            onnx_version="1.8.0",
            opset_version=11,
            providers=["CPUExecutionProvider"],
            python_version="3.7",
            gpu_support=False
        )
    }
    
    for version_key, spec in version_specs.items():
        print(f"🔧 Setting up Conda environment: {version_key}")
        print(f"   ONNX: {spec.onnx_version}, Python: {spec.python_version}")
        
        # Note: Này chỉ là demo - actual implementation cần conda installed
        try:
            # success = version_manager.create_conda_environment(version_key)
            # print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
            print("   Result: ⚠️  Skipped (demo mode)")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()


def demo_docker_containerization():
    """Demo Docker containerization"""
    print("🐳 Docker Containerization Demo")
    print("=" * 50)
    
    version_manager = ONNXVersionManager("./demo_onnx_environments")
    
    # Example Docker container setup
    gpu_spec = ONNXVersionSpec(
        onnx_version="1.12.0",
        opset_version=13,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        python_version="3.8",
        gpu_support=True
    )
    
    print("🔧 Setting up Docker container for GPU model")
    print(f"   ONNX: {gpu_spec.onnx_version}")
    print(f"   GPU Support: {gpu_spec.gpu_support}")
    print(f"   Providers: {gpu_spec.providers}")
    
    try:
        # container_name = version_manager.create_docker_container("gpu_model")
        # print(f"   Result: {'✅ Success' if container_name else '❌ Failed'}")
        print("   Result: ⚠️  Skipped (demo mode - requires Docker)")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()


def demo_version_compatibility_matrix():
    """Demo version compatibility matrix"""
    print("📊 Version Compatibility Matrix")
    print("=" * 50)
    
    # Model compatibility examples
    compatibility_matrix = [
        {
            "model": "BERT Text Classification",
            "framework": "PyTorch -> ONNX",
            "onnx_versions": ["1.12+", "1.15+"],
            "opset": "13-17",
            "issues": "None"
        },
        {
            "model": "ResNet50 Image Classification", 
            "framework": "TensorFlow -> ONNX",
            "onnx_versions": ["1.8+", "1.12+"],
            "opset": "11-13",
            "issues": "Opset 14+ có thể có compatibility issues"
        },
        {
            "model": "Legacy LSTM Model",
            "framework": "Keras -> ONNX",
            "onnx_versions": ["1.6", "1.8"],
            "opset": "9-11",
            "issues": "Không tương thích với ONNX 1.12+"
        },
        {
            "model": "YOLOv8 Object Detection",
            "framework": "Ultralytics -> ONNX",
            "onnx_versions": ["1.12+"],
            "opset": "13+",
            "issues": "Requires GPU providers cho optimal performance"
        }
    ]
    
    # Print compatibility table
    print(f"{'Model':<30} {'ONNX Versions':<15} {'Opset':<10} {'Issues':<40}")
    print("-" * 95)
    
    for item in compatibility_matrix:
        print(f"{item['model']:<30} {item['onnx_versions'][0]:<15} {item['opset']:<10} {item['issues']:<40}")
    
    print()


def demo_performance_comparison():
    """Demo performance comparison giữa các versions"""
    print("⚡ Performance Comparison Demo")
    print("=" * 50)
    
    # Mock performance data
    performance_data = {
        "bert-base-uncased": {
            "onnx_1.8": {"latency": "45ms", "memory": "800MB", "throughput": "22 req/s"},
            "onnx_1.12": {"latency": "35ms", "memory": "750MB", "throughput": "28 req/s"},
            "onnx_1.15": {"latency": "30ms", "memory": "700MB", "throughput": "33 req/s"},
        },
        "resnet50": {
            "onnx_1.8": {"latency": "25ms", "memory": "1.2GB", "throughput": "40 req/s"},
            "onnx_1.12": {"latency": "20ms", "memory": "1.1GB", "throughput": "50 req/s"},
            "onnx_1.15": {"latency": "18ms", "memory": "1.0GB", "throughput": "55 req/s"},
        }
    }
    
    for model, versions in performance_data.items():
        print(f"📈 {model}")
        print(f"{'Version':<12} {'Latency':<10} {'Memory':<10} {'Throughput':<12}")
        print("-" * 45)
        
        for version, metrics in versions.items():
            print(f"{version:<12} {metrics['latency']:<10} {metrics['memory']:<10} {metrics['throughput']:<12}")
        
        print()


def demo_real_world_scenarios():
    """Demo real-world usage scenarios"""
    print("🌍 Real-World Scenarios Demo")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "🏢 Enterprise ML Platform",
            "description": "Multiple teams với different model requirements",
            "solution": """
            - Kubernetes cluster với multiple namespaces
            - Mỗi team có riêng ONNX version requirements
            - Auto-scaling dựa trên load
            - Resource quotas và monitoring
            """
        },
        {
            "name": "🚀 Startup MVP",
            "description": "Fast iteration, limited resources",
            "solution": """
            - Conda environments cho development
            - Docker containers cho staging/production
            - Single server deployment
            - Manual scaling
            """
        },
        {
            "name": "🏭 Production ML API",
            "description": "High availability, consistent performance",
            "solution": """
            - Docker Swarm hoặc Kubernetes
            - Load balancing với health checks
            - Blue-green deployments
            - Monitoring và alerting
            """
        },
        {
            "name": "🧪 Research Environment",
            "description": "Experimentation với nhiều model versions",
            "solution": """
            - Multiple Conda environments
            - Easy switching between versions
            - Jupyter notebook integration
            - Version comparison tools
            """
        }
    ]
    
    for scenario in scenarios:
        print(f"{scenario['name']}")
        print(f"Problem: {scenario['description']}")
        print(f"Solution:{scenario['solution']}")
        print()


def main():
    """Main demo function"""
    print("🎯 ONNX Version Management System Demo")
    print("=" * 60)
    print()
    
    # Run all demos
    demo_version_management()
    demo_deployment_strategies()
    
    # Generate deployment files (commented out để không tạo files thật)
    # demo_deployment_file_generation()
    
    demo_conda_environment_setup()
    demo_docker_containerization()
    demo_version_compatibility_matrix()
    demo_performance_comparison()
    demo_real_world_scenarios()
    
    print("=" * 60)
    print("✅ Demo completed!")
    print()
    print("💡 Key Takeaways:")
    print("   1. Multiple ONNX versions can coexist using different strategies")
    print("   2. Choose deployment strategy based on requirements")
    print("   3. Conda: Good for development và isolated environments")
    print("   4. Docker: Good for consistent production deployments")
    print("   5. Kubernetes: Good for scalable, high-availability services")
    print("   6. Version management prevents compatibility issues")
    print("   7. Performance varies between ONNX versions")


if __name__ == "__main__":
    main() 