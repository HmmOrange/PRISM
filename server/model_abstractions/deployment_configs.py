"""
Deployment Configurations cho ONNX Version Management

Cung cấp các deployment strategies khác nhau để handle multiple ONNX versions:
1. Conda environments (isolated Python environments)
2. Docker containers (fully isolated environments)  
3. Kubernetes pods (scalable container orchestration)
4. Virtual environments (lightweight isolation)
5. Process isolation (subprocess-based)
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class DeploymentStrategy(Enum):
    """Các deployment strategies có sẵn"""
    CURRENT_ENV = "current_env"        # Sử dụng environment hiện tại
    CONDA = "conda"                    # Conda environments
    DOCKER = "docker"                  # Docker containers
    KUBERNETES = "kubernetes"          # K8s pods
    VENV = "venv"                     # Python virtual environments
    PROCESS = "process"               # Subprocess isolation


@dataclass
class ModelDeploymentConfig:
    """Configuration cho deployment của một model cụ thể"""
    model_id: str
    onnx_version: str
    opset_version: int
    strategy: DeploymentStrategy
    
    # Resource requirements
    memory_limit: str = "2Gi"
    cpu_limit: str = "1000m"
    gpu_required: bool = False
    gpu_memory: str = "1Gi"
    
    # Environment specific configs
    python_version: str = "3.9"
    providers: List[str] = None
    custom_packages: List[str] = None
    environment_variables: Dict[str, str] = None
    
    # Networking (for Docker/K8s)
    port: int = 8000
    health_check_path: str = "/health"
    
    # Scaling (for K8s)
    replicas: int = 1
    auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 5
    
    def __post_init__(self):
        if self.providers is None:
            self.providers = ["CPUExecutionProvider"]
        if self.custom_packages is None:
            self.custom_packages = []
        if self.environment_variables is None:
            self.environment_variables = {}


class DeploymentConfigManager:
    """Quản lý deployment configurations"""
    
    def __init__(self, config_file: str = "deployment_configs.yaml"):
        self.config_file = config_file
        self.configs: Dict[str, ModelDeploymentConfig] = {}
        self.global_settings: Dict[str, Any] = {}
        
        self.load_configurations()
    
    def load_configurations(self):
        """Load configurations từ YAML file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = yaml.safe_load(f)
                
                # Load global settings
                self.global_settings = data.get('global', {})
                
                # Load model configs
                for model_id, config_data in data.get('models', {}).items():
                    config_data['model_id'] = model_id
                    config_data['strategy'] = DeploymentStrategy(config_data['strategy'])
                    self.configs[model_id] = ModelDeploymentConfig(**config_data)
        else:
            self.create_default_configurations()
    
    def save_configurations(self):
        """Lưu configurations ra YAML file"""
        data = {
            'global': self.global_settings,
            'models': {}
        }
        
        for model_id, config in self.configs.items():
            config_dict = asdict(config)
            config_dict['strategy'] = config.strategy.value
            del config_dict['model_id']  # Remove model_id from nested dict
            data['models'][model_id] = config_dict
        
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def create_default_configurations(self):
        """Tạo default configurations"""
        
        # Global settings
        self.global_settings = {
            'default_strategy': DeploymentStrategy.CONDA.value,
            'base_image': 'python:3.9-slim',
            'docker_registry': 'localhost:5000',
            'namespace': 'onnx-models',
            'resource_limits': {
                'memory': '2Gi',
                'cpu': '1000m'
            }
        }
        
        # Example model configurations
        example_configs = {
            'bert-base-uncased': ModelDeploymentConfig(
                model_id='bert-base-uncased',
                onnx_version='1.15.0',
                opset_version=17,
                strategy=DeploymentStrategy.CONDA,
                memory_limit='4Gi',
                cpu_limit='2000m',
                providers=['CPUExecutionProvider']
            ),
            
            'resnet50-image-classifier': ModelDeploymentConfig(
                model_id='resnet50-image-classifier',
                onnx_version='1.12.0',
                opset_version=13,
                strategy=DeploymentStrategy.DOCKER,
                memory_limit='3Gi',
                gpu_required=True,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            ),
            
            'legacy-text-model': ModelDeploymentConfig(
                model_id='legacy-text-model',
                onnx_version='1.8.0',
                opset_version=11,
                strategy=DeploymentStrategy.VENV,
                python_version='3.7',
                providers=['CPUExecutionProvider']
            )
        }
        
        self.configs.update(example_configs)
        self.save_configurations()
    
    def get_config(self, model_id: str) -> Optional[ModelDeploymentConfig]:
        """Lấy configuration cho model"""
        return self.configs.get(model_id)
    
    def register_model_config(self, config: ModelDeploymentConfig):
        """Đăng ký configuration cho model"""
        self.configs[config.model_id] = config
        self.save_configurations()
    
    def get_deployment_template(self, strategy: DeploymentStrategy) -> Dict[str, Any]:
        """Lấy deployment template cho strategy"""
        
        templates = {
            DeploymentStrategy.CONDA: self._get_conda_template(),
            DeploymentStrategy.DOCKER: self._get_docker_template(),
            DeploymentStrategy.KUBERNETES: self._get_k8s_template(),
            DeploymentStrategy.VENV: self._get_venv_template(),
            DeploymentStrategy.PROCESS: self._get_process_template()
        }
        
        return templates.get(strategy, {})
    
    def _get_conda_template(self) -> Dict[str, Any]:
        """Template cho Conda deployment"""
        return {
            'environment_file': '''
name: onnx_{model_id}
channels:
  - conda-forge
  - defaults
dependencies:
  - python={python_version}
  - pip
  - pip:
    - onnxruntime=={onnx_version}
    - numpy
    - pillow
    - fastapi
    - uvicorn
    - {custom_packages}
''',
            'activation_script': '''
conda activate onnx_{model_id}
python -c "import onnxruntime; print(f'ONNX Runtime version: {{onnxruntime.__version__}}')"
''',
            'inference_script': '''
import onnxruntime as ort
import sys
import json
import numpy as np

# Load model
session = ort.InferenceSession(sys.argv[1], providers={providers})

# Inference function
def predict(inputs):
    # Convert inputs to numpy arrays
    input_feed = {{}}
    for name, data in inputs.items():
        input_feed[name] = np.array(data)
    
    # Run inference
    outputs = session.run(None, input_feed)
    
    # Convert to lists
    return [output.tolist() for output in outputs]

# Main loop for subprocess communication
if __name__ == "__main__":
    while True:
        try:
            line = input()
            if line.strip() == "EXIT":
                break
            
            inputs = json.loads(line)
            outputs = predict(inputs)
            print(json.dumps(outputs))
            sys.stdout.flush()
        except EOFError:
            break
        except Exception as e:
            print(json.dumps({{"error": str(e)}}))
            sys.stdout.flush()
'''
        }
    
    def _get_docker_template(self) -> Dict[str, Any]:
        """Template cho Docker deployment"""
        return {
            'dockerfile': '''
FROM {base_image}

# Install ONNX Runtime
RUN pip install onnxruntime{gpu_suffix}=={onnx_version}

# Install dependencies
RUN pip install numpy pillow fastapi uvicorn {custom_packages}

# Copy model files
COPY . /app
WORKDIR /app

# Set environment variables
{env_vars}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{port}{health_check_path} || exit 1

# Expose port
EXPOSE {port}

# Run inference server
CMD ["python", "inference_server.py"]
''',
            'compose_service': '''
version: '3.8'
services:
  onnx-{model_id}:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{port}:{port}"
    environment:
      {env_vars}
    deploy:
      resources:
        limits:
          memory: {memory_limit}
          cpus: '{cpu_limit}'
        reservations:
          devices:
            - driver: nvidia
              count: {gpu_count}
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}{health_check_path}"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
''',
            'inference_server': '''
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import os

app = FastAPI(title="ONNX Model Server - {model_id}")

# Load model
model_path = os.environ.get('MODEL_PATH', '/app/model.onnx')
session = ort.InferenceSession(model_path, providers={providers})

class PredictionRequest(BaseModel):
    inputs: Dict[str, Any]

class PredictionResponse(BaseModel):
    outputs: list
    model_id: str = "{model_id}"

@app.get("{health_check_path}")
async def health_check():
    return {{"status": "healthy", "model": "{model_id}"}}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert inputs to numpy arrays
        input_feed = {{}}
        for name, data in request.inputs.items():
            input_feed[name] = np.array(data)
        
        # Run inference
        outputs = session.run(None, input_feed)
        
        # Convert to lists
        result = [output.tolist() for output in outputs]
        
        return PredictionResponse(outputs=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''
        }
    
    def _get_k8s_template(self) -> Dict[str, Any]:
        """Template cho Kubernetes deployment"""
        return {
            'deployment': '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: onnx-{model_id}
  namespace: {namespace}
  labels:
    app: onnx-{model_id}
    version: {onnx_version}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: onnx-{model_id}
  template:
    metadata:
      labels:
        app: onnx-{model_id}
        version: {onnx_version}
    spec:
      containers:
      - name: onnx-runtime
        image: {docker_registry}/onnx-{model_id}:{onnx_version}
        ports:
        - containerPort: {port}
        env:
        {env_vars}
        resources:
          limits:
            memory: {memory_limit}
            cpu: {cpu_limit}
            nvidia.com/gpu: {gpu_count}
          requests:
            memory: {memory_request}
            cpu: {cpu_request}
        livenessProbe:
          httpGet:
            path: {health_check_path}
            port: {port}
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: {health_check_path}
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
      nodeSelector:
        kubernetes.io/arch: amd64
        {gpu_node_selector}
''',
            'service': '''
apiVersion: v1
kind: Service
metadata:
  name: onnx-{model_id}-service
  namespace: {namespace}
spec:
  selector:
    app: onnx-{model_id}
  ports:
  - protocol: TCP
    port: 80
    targetPort: {port}
  type: ClusterIP
''',
            'hpa': '''
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: onnx-{model_id}-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: onnx-{model_id}
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
        }
    
    def _get_venv_template(self) -> Dict[str, Any]:
        """Template cho Virtual Environment deployment"""
        return {
            'requirements_txt': '''
onnxruntime=={onnx_version}
numpy
pillow
fastapi
uvicorn
{custom_packages}
''',
            'setup_script': '''
#!/bin/bash
set -e

# Create virtual environment
python{python_version} -m venv venv_{model_id}

# Activate and install dependencies
source venv_{model_id}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment venv_{model_id} created successfully"
''',
            'run_script': '''
#!/bin/bash
source venv_{model_id}/bin/activate
python inference_server.py
'''
        }
    
    def _get_process_template(self) -> Dict[str, Any]:
        """Template cho Process isolation deployment"""
        return {
            'wrapper_script': '''
import subprocess
import json
import sys
import os
from typing import Dict, Any

class ProcessONNXWrapper:
    def __init__(self, model_path: str, python_executable: str = None):
        self.model_path = model_path
        self.python_executable = python_executable or sys.executable
        self.process = None
        self._start_process()
    
    def _start_process(self):
        """Start the inference process"""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        
        self.process = subprocess.Popen(
            [self.python_executable, 'inference_worker.py', self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
    
    def predict(self, inputs: Dict[str, Any]) -> list:
        """Run prediction"""
        if not self.process or self.process.poll() is not None:
            self._start_process()
        
        # Send inputs
        input_json = json.dumps(inputs)
        self.process.stdin.write(input_json + '\\n')
        self.process.stdin.flush()
        
        # Read outputs
        output_line = self.process.stdout.readline()
        if not output_line:
            raise Exception("Process died unexpectedly")
        
        return json.loads(output_line.strip())
    
    def close(self):
        """Close the process"""
        if self.process:
            self.process.stdin.write("EXIT\\n")
            self.process.stdin.flush()
            self.process.wait()
            self.process = None
''',
            'worker_script': '''
import onnxruntime as ort
import json
import sys
import numpy as np

# Load model
model_path = sys.argv[1]
session = ort.InferenceSession(model_path)

# Main loop
while True:
    try:
        line = input().strip()
        if line == "EXIT":
            break
        
        inputs = json.loads(line)
        
        # Convert to numpy arrays
        input_feed = {{}}
        for name, data in inputs.items():
            input_feed[name] = np.array(data)
        
        # Run inference
        outputs = session.run(None, input_feed)
        
        # Convert to lists and output
        result = [output.tolist() for output in outputs]
        print(json.dumps(result))
        sys.stdout.flush()
        
    except EOFError:
        break
    except Exception as e:
        error_result = {{"error": str(e)}}
        print(json.dumps(error_result))
        sys.stdout.flush()
'''
        }
    
    def generate_deployment_files(self, model_id: str, output_dir: str):
        """Generate deployment files cho model"""
        config = self.get_config(model_id)
        if not config:
            raise ValueError(f"No configuration found for model {model_id}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        template = self.get_deployment_template(config.strategy)
        
        # Template variables
        template_vars = {
            'model_id': model_id.replace('/', '-'),
            'onnx_version': config.onnx_version,
            'opset_version': config.opset_version,
            'python_version': config.python_version,
            'providers': config.providers,
            'custom_packages': ' '.join(config.custom_packages),
            'memory_limit': config.memory_limit,
            'cpu_limit': config.cpu_limit,
            'port': config.port,
            'health_check_path': config.health_check_path,
            'replicas': config.replicas,
            'min_replicas': config.min_replicas,
            'max_replicas': config.max_replicas,
            'env_vars': '\\n'.join([f'{k}={v}' for k, v in config.environment_variables.items()]),
            'gpu_count': 1 if config.gpu_required else 0,
            'gpu_suffix': '-gpu' if config.gpu_required else '',
            'namespace': self.global_settings.get('namespace', 'default'),
            'docker_registry': self.global_settings.get('docker_registry', 'localhost:5000'),
            'base_image': self.global_settings.get('base_image', 'python:3.9-slim'),
        }
        
        # Generate files based on strategy
        for filename, content in template.items():
            file_path = os.path.join(output_dir, filename)
            
            # Format template
            formatted_content = content.format(**template_vars)
            
            with open(file_path, 'w') as f:
                f.write(formatted_content)
        
        print(f"[ success ] Generated deployment files for {model_id} in {output_dir}")


# Helper function to auto-detect best deployment strategy
def recommend_deployment_strategy(
    model_complexity: str = "medium",  # simple, medium, complex
    expected_load: str = "low",        # low, medium, high
    isolation_required: bool = False,
    gpu_required: bool = False,
    development_mode: bool = True
) -> DeploymentStrategy:
    """Recommend deployment strategy dựa trên requirements"""
    
    if development_mode:
        if isolation_required:
            return DeploymentStrategy.CONDA
        else:
            return DeploymentStrategy.CURRENT_ENV
    
    if expected_load == "high":
        return DeploymentStrategy.KUBERNETES
    
    if gpu_required or model_complexity == "complex":
        return DeploymentStrategy.DOCKER
    
    if isolation_required:
        return DeploymentStrategy.CONDA
    
    if model_complexity == "simple":
        return DeploymentStrategy.VENV
    
    return DeploymentStrategy.DOCKER  # Safe default 