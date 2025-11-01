import os
import json
import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base_model import BaseModelWrapper, ModelFramework, ModelType, ModelRegistry
from .model_converter import AutoConverter


class ModelManager:
    """
    Model Manager để quản lý lifecycle của các models trong hệ thống
    """
    
    def __init__(
        self,
        max_models_in_ram: int = 5,
        max_models_in_disk: int = 10,
        models_directory: str = "models",
        auto_convert_to_onnx: bool = False
    ):
        self.max_models_in_ram = max_models_in_ram
        self.max_models_in_disk = max_models_in_disk
        self.models_directory = models_directory
        self.auto_convert_to_onnx = auto_convert_to_onnx
        
        # Registry để track các models đang được quản lý
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Tạo models directory nếu chưa có
        os.makedirs(models_directory, exist_ok=True)
    
    def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        framework: Optional[ModelFramework] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Đăng ký một model vào manager
        
        Args:
            model_id: Unique identifier cho model
            model_type: Loại model
            framework: Framework của model (auto-detect nếu None)
            metadata: Metadata bổ sung cho model
            **kwargs: Additional arguments cho wrapper
        
        Returns:
            bool: True nếu đăng ký thành công
        """
        try:
            model_path = os.path.join(self.models_directory, model_id)
            
            # Auto-detect framework nếu không có
            if framework is None:
                framework = AutoConverter.detect_model_framework(model_path)
                if framework is None:
                    print(f"[ error ] Cannot detect framework for model: {model_id}")
                    return False
            
            # Check if model exists on disk
            model_exists_on_disk = os.path.exists(model_path) and os.listdir(model_path)
            lasted_used_in_disk = datetime.datetime.now() if model_exists_on_disk else None
            
            self.models[model_id] = {
                "wrapper": None,
                "model_type": model_type,
                "framework": framework,
                "model_path": model_path,
                "metadata": metadata or {},
                "kwargs": kwargs,
                "lasted_used": None,
                "lasted_used_in_disk": lasted_used_in_disk,
                "is_using": False
            }
            
            print(f"[ success ] Model registered: {model_id} ({framework.value})")
            return True
            
        except Exception as e:
            print(f"[ error ] Failed to register model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str, device: str = "cpu") -> Optional[BaseModelWrapper]:
        """
        Load model vào memory
        
        Args:
            model_id: ID của model cần load
            device: Device để load model
        
        Returns:
            BaseModelWrapper: Model wrapper nếu load thành công
        """
        if model_id not in self.models:
            print(f"[ error ] Model not registered: {model_id}")
            return None
        
        model_info = self.models[model_id]
        
        # Nếu model đã được load, update timestamp và return
        if model_info["wrapper"] is not None:
            model_info["lasted_used"] = datetime.datetime.now()
            model_info["lasted_used_in_disk"] = datetime.datetime.now()
            return model_info["wrapper"]
        
        try:
            # Manage RAM - unload oldest model nếu cần
            self._manage_ram_usage()
            
            # Download model nếu cần
            if not self._ensure_model_downloaded(model_id):
                return None
            
            # Manage disk space sau khi download
            self._manage_disk_usage()
            
            # Auto convert to ONNX nếu enabled
            if self.auto_convert_to_onnx and model_info["framework"] != ModelFramework.ONNX:
                self._try_convert_to_onnx(model_id)
            
            # Create model wrapper
            wrapper = ModelRegistry.create_model_wrapper(
                model_path=model_info["model_path"],
                model_type=model_info["model_type"],
                framework=model_info["framework"],
                device=device,
                metadata=model_info["metadata"],
                **model_info["kwargs"]
            )
            
            if wrapper is None:
                print(f"[ error ] No wrapper found for model {model_id}")
                return None
            
            # Load model
            wrapper.load_model()
            
            # Update tracking info
            model_info["wrapper"] = wrapper
            model_info["lasted_used"] = datetime.datetime.now()
            model_info["lasted_used_in_disk"] = datetime.datetime.now()
            
            print(f"[ success ] Model loaded: {model_id}")
            return wrapper
            
        except Exception as e:
            print(f"[ error ] Failed to load model {model_id}: {e}")
            return None
    
    def unload_model(self, model_id: str) -> bool:
        """Unload model khỏi memory"""
        if model_id not in self.models:
            return False
        
        model_info = self.models[model_id]
        if model_info["wrapper"] is not None:
            model_info["wrapper"].unload_model()
            model_info["wrapper"] = None
            model_info["lasted_used"] = None
            print(f"[ success ] Model unloaded: {model_id}")
        
        return True
    
    def get_model(self, model_id: str, device: str = "cpu") -> Optional[BaseModelWrapper]:
        """Get model (load nếu chưa có trong memory)"""
        return self.load_model(model_id, device)
    
    def predict(
        self, 
        model_id: str, 
        inputs: Dict[str, Any], 
        device: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """
        Thực hiện prediction với model
        
        Args:
            model_id: ID của model
            inputs: Input data
            device: Device để chạy inference
        
        Returns:
            Dict: Kết quả prediction
        """
        wrapper = self.get_model(model_id, device)
        if wrapper is None:
            return None
        
        try:
            # Mark as using
            self.models[model_id]["is_using"] = True
            
            # Run prediction
            result = wrapper.predict(inputs)
            
            return result
            
        except Exception as e:
            print(f"[ error ] Prediction failed for model {model_id}: {e}")
            return None
        
        finally:
            # Unmark as using
            if model_id in self.models:
                self.models[model_id]["is_using"] = False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Liệt kê tất cả models được quản lý"""
        models_info = []
        for model_id, info in self.models.items():
            models_info.append({
                "model_id": model_id,
                "model_type": info["model_type"].value,
                "framework": info["framework"].value,
                "is_loaded": info["wrapper"] is not None,
                "is_using": info["is_using"],
                "lasted_used": info["lasted_used"],
                "lasted_used_in_disk": info["lasted_used_in_disk"],
                "metadata": info["metadata"]
            })
        return models_info
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một model"""
        if model_id not in self.models:
            return None
        
        info = self.models[model_id]
        model_info = {
            "model_id": model_id,
            "model_type": info["model_type"].value,
            "framework": info["framework"].value,
            "model_path": info["model_path"],
            "is_loaded": info["wrapper"] is not None,
            "is_using": info["is_using"],
            "lasted_used": info["lasted_used"],
            "lasted_used_in_disk": info["lasted_used_in_disk"],
            "metadata": info["metadata"]
        }
        
        if info["wrapper"] is not None:
            model_info.update(info["wrapper"].get_model_info())
        
        return model_info
    
    def _manage_ram_usage(self):
        """Quản lý RAM usage - unload oldest models"""
        loaded_models = [
            model_id for model_id, info in self.models.items()
            if info["wrapper"] is not None and not info["is_using"]
        ]
        
        while len(loaded_models) >= self.max_models_in_ram:
            # Find oldest model to unload
            oldest_model_id = min(
                loaded_models,
                key=lambda x: self.models[x]["lasted_used"] or datetime.datetime.min
            )
            
            print(f"[ cleanup ram ] Unloading model {oldest_model_id}")
            self.unload_model(oldest_model_id)
            loaded_models.remove(oldest_model_id)
    
    def _manage_disk_usage(self):
        """Quản lý disk usage - delete oldest models"""
        models_on_disk = [
            model_id for model_id, info in self.models.items()
            if info["lasted_used_in_disk"] is not None
        ]
        
        while len(models_on_disk) > self.max_models_in_disk:
            # Prioritize models not in RAM for deletion
            models_not_in_ram = [
                model_id for model_id in models_on_disk
                if self.models[model_id]["wrapper"] is None
            ]
            
            if models_not_in_ram:
                oldest_model_id = min(
                    models_not_in_ram,
                    key=lambda x: self.models[x]["lasted_used_in_disk"]
                )
            else:
                oldest_model_id = min(
                    models_on_disk,
                    key=lambda x: self.models[x]["lasted_used_in_disk"]
                )
            
            print(f"[ cleanup disk ] Deleting model {oldest_model_id}")
            self._delete_model_from_disk(oldest_model_id)
            models_on_disk.remove(oldest_model_id)
    
    def _ensure_model_downloaded(self, model_id: str) -> bool:
        """Đảm bảo model đã được download"""
        model_path = self.models[model_id]["model_path"]
        
        if os.path.exists(model_path) and os.listdir(model_path):
            return True
        
        # Download logic - tùy theo framework
        # Ví dụ cho HuggingFace
        if self.models[model_id]["framework"] == ModelFramework.HUGGINGFACE:
            return self._download_huggingface_model(model_id)
        
        print(f"[ error ] Download not implemented for framework: {self.models[model_id]['framework']}")
        return False
    
    def _download_huggingface_model(self, model_id: str) -> bool:
        """Download HuggingFace model từ Hub"""
        try:
            model_path = self.models[model_id]["model_path"]
            os.makedirs(model_path, exist_ok=True)
            
            # Git clone từ HuggingFace Hub
            clone_cmd = f"git clone --recurse-submodules https://huggingface.co/{model_id} {model_path}"
            result = os.system(clone_cmd)
            
            if result != 0:
                print(f"[ error ] Failed to download model {model_id}")
                # Clean up
                import shutil
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                return False
            
            # Update disk tracking
            self.models[model_id]["lasted_used_in_disk"] = datetime.datetime.now()
            print(f"[ success ] Downloaded model {model_id}")
            return True
            
        except Exception as e:
            print(f"[ error ] Failed to download model {model_id}: {e}")
            return False
    
    def _delete_model_from_disk(self, model_id: str):
        """Xóa model khỏi disk"""
        try:
            # Unload from RAM first
            self.unload_model(model_id)
            
            # Delete folder
            model_path = self.models[model_id]["model_path"]
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
                print(f"[ deleted ] Model folder {model_path}")
            
            # Reset disk tracking
            self.models[model_id]["lasted_used_in_disk"] = None
            
        except Exception as e:
            print(f"[ warning ] Failed to delete model {model_id}: {e}")
    
    def _try_convert_to_onnx(self, model_id: str):
        """Thử convert model sang ONNX format"""
        try:
            model_info = self.models[model_id]
            
            if model_info["framework"] == ModelFramework.ONNX:
                return  # Already ONNX
            
            onnx_path = f"{model_info['model_path']}_onnx"
            
            success = AutoConverter.auto_convert_to_onnx(
                model_info["model_path"],
                onnx_path,
                model_info["model_type"],
                model_info["framework"]
            )
            
            if success:
                # Update model info to use ONNX version
                model_info["framework"] = ModelFramework.ONNX
                model_info["model_path"] = onnx_path
                print(f"[ success ] Model {model_id} converted to ONNX")
            
        except Exception as e:
            print(f"[ warning ] Failed to convert model {model_id} to ONNX: {e}") 