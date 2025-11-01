from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import numpy as np


class ModelFramework(Enum):
    """Enum định nghĩa các framework được hỗ trợ"""
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    KERAS = "keras"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    YOLO = "yolo"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class ModelType(Enum):
    """Enum định nghĩa các loại task được hỗ trợ"""
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question-answering"
    TEXT_GENERATION = "text-generation"
    SENTENCE_SIMILARITY = "sentence-similarity"
    
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_TO_TEXT = "image-to-text"
    
    AUDIO_CLASSIFICATION = "audio-classification"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    
    TABULAR_CLASSIFICATION = "tabular-classification"
    TABULAR_REGRESSION = "tabular-regression"
    
    VIDEO_CLASSIFICATION = "video-classification"


class BaseModelWrapper(ABC):
    """
    Lớp trừu tượng cơ bản cho tất cả các model wrapper.
    Cung cấp interface thống nhất cho việc inference.
    """
    
    def __init__(
        self, 
        model_path: str, 
        model_type: ModelType,
        framework: ModelFramework,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.framework = framework
        self.device = device
        self.metadata = metadata or {}
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model vào memory"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload model khỏi memory"""
        pass
    
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực hiện inference với inputs được cung cấp
        
        Args:
            inputs: Dictionary chứa input data
            
        Returns:
            Dictionary chứa kết quả prediction
        """
        pass
    
    @abstractmethod
    def to_device(self, device: str) -> 'BaseModelWrapper':
        """Chuyển model sang device khác"""
        pass
    
    def is_loaded(self) -> bool:
        """Kiểm tra xem model đã được load chưa"""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Trả về thông tin về model"""
        return {
            "model_path": self.model_path,
            "model_type": self.model_type.value,
            "framework": self.framework.value,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "metadata": self.metadata
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs trước khi inference"""
        # Override trong subclass nếu cần validation đặc biệt
        return True


class ModelRegistry:
    """
    Registry để quản lý các model wrapper classes
    """
    
    _registry: Dict[ModelFramework, Dict[ModelType, type]] = {}
    
    @classmethod
    def register(
        cls, 
        framework: ModelFramework, 
        model_type: ModelType, 
        wrapper_class: type
    ):
        """Đăng ký một wrapper class cho framework và model type cụ thể"""
        if framework not in cls._registry:
            cls._registry[framework] = {}
        cls._registry[framework][model_type] = wrapper_class
    
    @classmethod
    def get_wrapper_class(
        cls, 
        framework: ModelFramework, 
        model_type: ModelType
    ) -> Optional[type]:
        """Lấy wrapper class cho framework và model type"""
        return cls._registry.get(framework, {}).get(model_type)
    
    @classmethod
    def create_model_wrapper(
        cls,
        model_path: str,
        model_type: ModelType,
        framework: ModelFramework,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[BaseModelWrapper]:
        """
        Factory method để tạo model wrapper
        """
        wrapper_class = cls.get_wrapper_class(framework, model_type)
        if wrapper_class is None:
            raise ValueError(
                f"No wrapper found for framework {framework.value} "
                f"and model type {model_type.value}"
            )
        
        return wrapper_class(
            model_path=model_path,
            model_type=model_type,
            framework=framework,
            device=device,
            metadata=metadata,
            **kwargs
        )
    
    @classmethod
    def list_supported_combinations(cls) -> List[Dict[str, str]]:
        """Liệt kê tất cả các combination được hỗ trợ"""
        combinations = []
        for framework, model_types in cls._registry.items():
            for model_type in model_types:
                combinations.append({
                    "framework": framework.value,
                    "model_type": model_type.value
                })
        return combinations 