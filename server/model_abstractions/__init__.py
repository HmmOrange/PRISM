"""
Model Abstractions Package

Cung cấp lớp trừu tượng thống nhất cho việc quản lý và sử dụng các model 
từ nhiều framework khác nhau (HuggingFace, ONNX, PyTorch, TensorFlow, etc.)
"""

from .base_model import (
    BaseModelWrapper,
    ModelFramework,
    ModelType,
    ModelRegistry
)

from .onnx_wrapper import ONNXModelWrapper
from .huggingface_wrapper import HuggingFaceWrapper
from .specialized_wrappers import YOLOWrapper, TabularWrapper
from .model_converter import ModelConverter, AutoConverter

# Import để auto-register các wrappers
from . import onnx_wrapper
from . import huggingface_wrapper
from . import specialized_wrappers

__all__ = [
    # Base classes
    'BaseModelWrapper',
    'ModelFramework',
    'ModelType',
    'ModelRegistry',
    
    # Wrapper implementations
    'ONNXModelWrapper',
    'HuggingFaceWrapper',
    'YOLOWrapper',
    'TabularWrapper',
    
    # Utilities
    'ModelConverter',
    'AutoConverter',
]

# Version info
__version__ = "1.0.0" 