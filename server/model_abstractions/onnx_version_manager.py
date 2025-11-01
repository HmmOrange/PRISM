import os
import json
import subprocess
import sys
import importlib
import importlib.util
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import docker
from concurrent.futures import ThreadPoolExecutor
import threading

from .base_model import BaseModelWrapper, ModelFramework, ModelType


@dataclass
class ONNXVersionSpec:
    """Specification cho ONNX Runtime version"""
    onnx_version: str           # e.g., "1.15.0"
    opset_version: int          # e.g., 11, 13, 17
    providers: List[str]        # e.g., ["CUDAExecutionProvider", "CPUExecutionProvider"]
    python_version: str = "3.9" # Python version compatibility
    gpu_support: bool = True    # Có cần GPU support không
    custom_ops: List[str] = None # Custom operators nếu có


class ONNXVersionManager:
    """
    Quản lý multiple versions của ONNX Runtime và model compatibility
    """
    
    def __init__(self, base_path: str = "./onnx_environments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Version specifications registry
        self.version_specs: Dict[str, ONNXVersionSpec] = {}
        self.model_version_mapping: Dict[str, str] = {}  # model_id -> version_key
        
        # Runtime environments
        self.runtime_environments: Dict[str, Any] = {}
        self.docker_client = None
        self.containers: Dict[str, Any] = {}
        
        # Load configurations
        self._load_version_configurations()
        self._init_docker_client()
    
    def _init_docker_client(self):
        """Initialize Docker client nếu có"""
        try:
            self.docker_client = docker.from_env()
            print("[ success ] Docker client initialized")
        except Exception as e:
            print(f"[ warning ] Docker not available: {e}")
            self.docker_client = None
    
    def _load_version_configurations(self):
        """Load version configurations từ file"""
        config_file = self.base_path / "onnx_versions.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                
                for version_key, spec_data in config.get("versions", {}).items():
                    self.version_specs[version_key] = ONNXVersionSpec(**spec_data)
                
                self.model_version_mapping = config.get("model_mappings", {})
        else:
            # Create default configurations
            self._create_default_configurations()
    
    def _create_default_configurations(self):
        """Tạo default version configurations"""
        
        # Common ONNX versions với compatibility
        default_versions = {
            "onnx_1_15": ONNXVersionSpec(
                onnx_version="1.15.0",
                opset_version=17,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                python_version="3.9"
            ),
            "onnx_1_12": ONNXVersionSpec(
                onnx_version="1.12.0", 
                opset_version=13,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                python_version="3.8"
            ),
            "onnx_1_8": ONNXVersionSpec(
                onnx_version="1.8.0",
                opset_version=11,
                providers=["CPUExecutionProvider"],
                python_version="3.7",
                gpu_support=False
            ),
            "onnx_legacy": ONNXVersionSpec(
                onnx_version="1.6.0",
                opset_version=10,
                providers=["CPUExecutionProvider"],
                python_version="3.7",
                gpu_support=False
            )
        }
        
        self.version_specs.update(default_versions)
        self._save_configurations()
    
    def _save_configurations(self):
        """Lưu configurations ra file"""
        config_file = self.base_path / "onnx_versions.json"
        
        config = {
            "versions": {
                key: {
                    "onnx_version": spec.onnx_version,
                    "opset_version": spec.opset_version,
                    "providers": spec.providers,
                    "python_version": spec.python_version,
                    "gpu_support": spec.gpu_support,
                    "custom_ops": spec.custom_ops or []
                }
                for key, spec in self.version_specs.items()
            },
            "model_mappings": self.model_version_mapping
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def register_model_version_requirement(
        self, 
        model_id: str, 
        version_spec: ONNXVersionSpec,
        version_key: Optional[str] = None
    ):
        """Đăng ký version requirement cho model"""
        
        if version_key is None:
            version_key = f"custom_{model_id}_{version_spec.onnx_version}"
        
        self.version_specs[version_key] = version_spec
        self.model_version_mapping[model_id] = version_key
        self._save_configurations()
        
        print(f"[ registered ] Model {model_id} requires ONNX {version_spec.onnx_version}")
    
    def get_version_for_model(self, model_id: str) -> Optional[ONNXVersionSpec]:
        """Lấy version specification cho model"""
        version_key = self.model_version_mapping.get(model_id)
        if version_key:
            return self.version_specs.get(version_key)
        
        # Default fallback
        return self.version_specs.get("onnx_1_15")
    
    def create_conda_environment(self, version_key: str) -> bool:
        """Tạo Conda environment cho ONNX version"""
        try:
            spec = self.version_specs[version_key]
            env_name = f"onnx_{version_key}"
            env_path = self.base_path / env_name
            
            if env_path.exists():
                print(f"[ exists ] Conda environment {env_name} already exists")
                return True
            
            # Create conda environment
            conda_cmd = [
                "conda", "create", "-n", env_name, 
                f"python={spec.python_version}", "-y"
            ]
            subprocess.run(conda_cmd, check=True)
            
            # Install ONNX Runtime
            if spec.gpu_support:
                pip_cmd = [
                    "conda", "run", "-n", env_name, "pip", "install", 
                    f"onnxruntime-gpu=={spec.onnx_version}"
                ]
            else:
                pip_cmd = [
                    "conda", "run", "-n", env_name, "pip", "install",
                    f"onnxruntime=={spec.onnx_version}"
                ]
            
            subprocess.run(pip_cmd, check=True)
            
            # Install additional dependencies
            deps_cmd = [
                "conda", "run", "-n", env_name, "pip", "install",
                "numpy", "pillow", "torch", "transformers"
            ]
            subprocess.run(deps_cmd, check=True)
            
            print(f"[ success ] Created conda environment {env_name}")
            return True
            
        except Exception as e:
            print(f"[ error ] Failed to create conda environment: {e}")
            return False
    
    def create_docker_container(self, version_key: str) -> Optional[str]:
        """Tạo Docker container cho ONNX version"""
        if not self.docker_client:
            return None
        
        try:
            spec = self.version_specs[version_key]
            container_name = f"onnx_{version_key}"
            
            # Check if container exists
            if container_name in self.containers:
                return container_name
            
            # Create Dockerfile content
            dockerfile_content = self._generate_dockerfile(spec)
            
            # Build image
            image_tag = f"onnx_runtime_{version_key}"
            
            # Build context
            build_path = self.base_path / f"docker_{version_key}"
            build_path.mkdir(exist_ok=True)
            
            dockerfile_path = build_path / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build image
            image, build_logs = self.docker_client.images.build(
                path=str(build_path),
                tag=image_tag,
                rm=True
            )
            
            # Create container
            container = self.docker_client.containers.create(
                image_tag,
                name=container_name,
                detach=True,
                volumes={
                    str(self.base_path): {'bind': '/models', 'mode': 'rw'}
                },
                ports={'8000/tcp': None}  # Random port
            )
            
            self.containers[container_name] = container
            
            print(f"[ success ] Created Docker container {container_name}")
            return container_name
            
        except Exception as e:
            print(f"[ error ] Failed to create Docker container: {e}")
            return None
    
    def _generate_dockerfile(self, spec: ONNXVersionSpec) -> str:
        """Generate Dockerfile cho ONNX version"""
        
        base_image = f"python:{spec.python_version}-slim"
        
        if spec.gpu_support:
            base_image = f"nvidia/cuda:11.8-runtime-ubuntu20.04"
            python_install = f"""
RUN apt-get update && apt-get install -y python{spec.python_version} python3-pip
RUN ln -s /usr/bin/python{spec.python_version} /usr/bin/python
"""
        else:
            python_install = ""
        
        onnx_package = "onnxruntime-gpu" if spec.gpu_support else "onnxruntime"
        
        dockerfile = f"""
FROM {base_image}

{python_install}

WORKDIR /app

# Install ONNX Runtime
RUN pip install {onnx_package}=={spec.onnx_version}

# Install dependencies
RUN pip install numpy pillow torch transformers fastapi uvicorn

# Copy inference server
COPY inference_server.py /app/

EXPOSE 8000

CMD ["python", "inference_server.py"]
"""
        
        return dockerfile
    
    def get_runtime_session(self, model_id: str, model_path: str):
        """Lấy ONNX Runtime session với correct version"""
        
        spec = self.get_version_for_model(model_id)
        if not spec:
            raise ValueError(f"No version specification found for model {model_id}")
        
        version_key = self.model_version_mapping.get(model_id, "onnx_1_15")
        
        # Try different deployment methods
        session = self._get_conda_session(version_key, model_path)
        if session:
            return session
        
        session = self._get_docker_session(version_key, model_path)
        if session:
            return session
        
        # Fallback to current environment
        return self._get_current_env_session(model_path, spec)
    
    def _get_conda_session(self, version_key: str, model_path: str):
        """Get session từ Conda environment"""
        try:
            env_name = f"onnx_{version_key}"
            
            # Check if environment exists
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True, text=True, check=True
            )
            envs = json.loads(result.stdout)
            env_paths = [env for env in envs["envs"] if env_name in env]
            
            if not env_paths:
                # Create environment if not exists
                if not self.create_conda_environment(version_key):
                    return None
            
            # Import onnxruntime from conda env
            conda_python = f"conda run -n {env_name} python"
            
            # Use subprocess to run inference in conda env
            return CondaONNXSession(env_name, model_path)
            
        except Exception as e:
            print(f"[ error ] Failed to get conda session: {e}")
            return None
    
    def _get_docker_session(self, version_key: str, model_path: str):
        """Get session từ Docker container"""
        try:
            container_name = self.create_docker_container(version_key)
            if not container_name:
                return None
            
            container = self.containers[container_name]
            
            # Start container if not running
            if container.status != 'running':
                container.start()
            
            return DockerONNXSession(container, model_path)
            
        except Exception as e:
            print(f"[ error ] Failed to get docker session: {e}")
            return None
    
    def _get_current_env_session(self, model_path: str, spec: ONNXVersionSpec):
        """Fallback to current environment session"""
        try:
            import onnxruntime as ort
            
            # Check version compatibility
            current_version = ort.__version__
            if current_version != spec.onnx_version:
                print(f"[ warning ] Version mismatch: current={current_version}, required={spec.onnx_version}")
            
            session = ort.InferenceSession(model_path, providers=spec.providers)
            return session
            
        except Exception as e:
            print(f"[ error ] Failed to create current env session: {e}")
            return None


class CondaONNXSession:
    """ONNX Session wrapper cho Conda environment"""
    
    def __init__(self, env_name: str, model_path: str):
        self.env_name = env_name
        self.model_path = model_path
        self._session_id = None
        self._init_session()
    
    def _init_session(self):
        """Initialize session trong conda environment"""
        # Create a simple inference server script
        script_content = f"""
import onnxruntime as ort
import json
import sys
import numpy as np
from pathlib import Path

session = ort.InferenceSession('{self.model_path}')

# Read input from stdin
input_data = json.loads(sys.stdin.read())

# Convert to numpy arrays
inputs = {{}}
for name, data in input_data.items():
    inputs[name] = np.array(data)

# Run inference
outputs = session.run(None, inputs)

# Convert outputs to lists and return
result = [output.tolist() for output in outputs]
print(json.dumps(result))
"""
        
        # Save script to temp file
        script_path = Path(f"/tmp/onnx_inference_{self.env_name}.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        self.script_path = script_path
    
    def run(self, output_names, input_feed):
        """Run inference trong conda environment"""
        try:
            # Prepare input data
            input_data = {}
            for name, data in input_feed.items():
                if hasattr(data, 'tolist'):
                    input_data[name] = data.tolist()
                else:
                    input_data[name] = data
            
            # Run inference via subprocess
            cmd = ["conda", "run", "-n", self.env_name, "python", str(self.script_path)]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=json.dumps(input_data))
            
            if process.returncode != 0:
                raise Exception(f"Inference failed: {stderr}")
            
            # Parse outputs
            outputs = json.loads(stdout)
            return [np.array(output) for output in outputs]
            
        except Exception as e:
            print(f"[ error ] Conda inference failed: {e}")
            raise


class DockerONNXSession:
    """ONNX Session wrapper cho Docker container"""
    
    def __init__(self, container, model_path: str):
        self.container = container
        self.model_path = model_path
        self.api_url = None
        self._start_inference_server()
    
    def _start_inference_server(self):
        """Start inference server trong container"""
        try:
            # Create inference server script
            server_script = """
import onnxruntime as ort
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

app = FastAPI()

# Load model
session = ort.InferenceSession('/models/model.onnx')

class InferenceRequest(BaseModel):
    inputs: Dict[str, Any]

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        # Convert inputs to numpy
        input_feed = {}
        for name, data in request.inputs.items():
            input_feed[name] = np.array(data)
        
        # Run inference
        outputs = session.run(None, input_feed)
        
        # Convert to lists
        result = [output.tolist() for output in outputs]
        
        return {"outputs": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
            
            # Copy script to container
            self.container.exec_run(
                f"sh -c 'cat > /app/inference_server.py << EOF\n{server_script}\nEOF'"
            )
            
            # Get container port
            port_info = self.container.attrs['NetworkSettings']['Ports']['8000/tcp']
            if port_info:
                host_port = port_info[0]['HostPort']
                self.api_url = f"http://localhost:{host_port}"
            
            print(f"[ success ] Docker inference server started at {self.api_url}")
            
        except Exception as e:
            print(f"[ error ] Failed to start Docker inference server: {e}")
    
    def run(self, output_names, input_feed):
        """Run inference via Docker API"""
        try:
            import requests
            
            if not self.api_url:
                raise Exception("Inference server not available")
            
            # Prepare request
            input_data = {}
            for name, data in input_feed.items():
                if hasattr(data, 'tolist'):
                    input_data[name] = data.tolist()
                else:
                    input_data[name] = data
            
            # Send request
            response = requests.post(
                f"{self.api_url}/predict",
                json={"inputs": input_data},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.text}")
            
            result = response.json()
            outputs = result["outputs"]
            
            return [np.array(output) for output in outputs]
            
        except Exception as e:
            print(f"[ error ] Docker inference failed: {e}")
            raise


# Integration với existing ONNX wrapper
class VersionedONNXWrapper(BaseModelWrapper):
    """ONNX Wrapper với version management"""
    
    def __init__(
        self,
        model_path: str,
        model_type: ModelType,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        version_manager: Optional[ONNXVersionManager] = None,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            framework=ModelFramework.ONNX,
            device=device,
            metadata=metadata
        )
        
        self.version_manager = version_manager or ONNXVersionManager()
        self.session = None
        self.model_id = kwargs.get('model_id', 'unknown')
    
    def load_model(self) -> None:
        """Load model với correct ONNX version"""
        try:
            onnx_model_path = f"{self.model_path}/model.onnx"
            
            # Get session với correct version
            self.session = self.version_manager.get_runtime_session(
                self.model_id, onnx_model_path
            )
            
            self._is_loaded = True
            print(f"[ success ] Versioned ONNX model loaded: {self.model_path}")
            
        except Exception as e:
            print(f"[ error ] Failed to load versioned ONNX model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload model"""
        if self.session:
            self.session = None
            self._is_loaded = False
            print(f"[ success ] Versioned ONNX model unloaded: {self.model_path}")
    
    def to_device(self, device: str) -> 'VersionedONNXWrapper':
        """Device transfer (handled by version manager)"""
        self.device = device
        return self
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Predict với versioned session"""
        if not self._is_loaded:
            self.load_model()
        
        # Preprocessing (same as before)
        processed_inputs = self._preprocess_inputs(inputs)
        
        # Run inference với versioned session
        input_names = [input.name for input in self.session.get_inputs()] if hasattr(self.session, 'get_inputs') else list(processed_inputs.keys())
        output_names = [output.name for output in self.session.get_outputs()] if hasattr(self.session, 'get_outputs') else None
        
        outputs = self.session.run(output_names, processed_inputs)
        
        # Postprocessing (same as before)
        return self._postprocess_outputs(outputs)
    
    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess inputs (implement from original ONNX wrapper)"""
        # Same logic as ONNXModelWrapper
        return inputs
    
    def _postprocess_outputs(self, outputs) -> Dict[str, Any]:
        """Postprocess outputs (implement from original ONNX wrapper)"""
        # Same logic as ONNXModelWrapper
        return {"output": outputs} 