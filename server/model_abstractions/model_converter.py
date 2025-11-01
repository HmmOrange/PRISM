import os
import json
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .base_model import ModelFramework, ModelType


class ModelConverter:
    """
    Utility class để convert models từ các framework khác nhau sang ONNX
    """
    
    @staticmethod
    def convert_huggingface_to_onnx(
        model_path: str,
        output_path: str,
        model_type: ModelType,
        sample_inputs: Optional[Dict[str, Any]] = None,
        opset_version: int = 11
    ) -> bool:
        """
        Convert HuggingFace model sang ONNX
        
        Args:
            model_path: Đường dẫn tới HuggingFace model
            output_path: Đường dẫn output cho ONNX model
            model_type: Loại model
            sample_inputs: Sample inputs để trace model
            opset_version: ONNX opset version
        
        Returns:
            bool: True nếu conversion thành công
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch.onnx
            
            # Load model và tokenizer
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            model.eval()
            
            # Tạo sample inputs nếu không có
            if sample_inputs is None:
                sample_inputs = ModelConverter._create_sample_inputs(model_type, tokenizer)
            
            # Tạo output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Convert sang ONNX
            onnx_path = os.path.join(output_path, "model.onnx")
            
            torch.onnx.export(
                model,
                tuple(sample_inputs.values()),
                onnx_path,
                input_names=list(sample_inputs.keys()),
                output_names=["output"],
                dynamic_axes={
                    name: {0: "batch_size", 1: "sequence_length"} 
                    for name in sample_inputs.keys()
                },
                opset_version=opset_version,
                do_constant_folding=True
            )
            
            # Lưu tokenizer config cho preprocessing
            ModelConverter._save_preprocessing_config(
                output_path, model_type, tokenizer
            )
            
            print(f"[ success ] HuggingFace model converted to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            print(f"[ error ] Failed to convert HuggingFace model to ONNX: {e}")
            return False
    
    @staticmethod
    def convert_pytorch_to_onnx(
        model: torch.nn.Module,
        output_path: str,
        model_type: ModelType,
        sample_inputs: torch.Tensor,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 11
    ) -> bool:
        """
        Convert PyTorch model sang ONNX
        """
        try:
            import torch.onnx
            
            model.eval()
            
            # Tạo output directory
            os.makedirs(output_path, exist_ok=True)
            onnx_path = os.path.join(output_path, "model.onnx")
            
            input_names = input_names or ["input"]
            output_names = output_names or ["output"]
            
            torch.onnx.export(
                model,
                sample_inputs,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                do_constant_folding=True
            )
            
            print(f"[ success ] PyTorch model converted to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            print(f"[ error ] Failed to convert PyTorch model to ONNX: {e}")
            return False
    
    @staticmethod
    def convert_tensorflow_to_onnx(
        model_path: str,
        output_path: str,
        model_type: ModelType
    ) -> bool:
        """
        Convert TensorFlow model sang ONNX
        """
        try:
            import tf2onnx
            import tensorflow as tf
            
            # Load TensorFlow model
            model = tf.saved_model.load(model_path)
            
            # Tạo output directory
            os.makedirs(output_path, exist_ok=True)
            onnx_path = os.path.join(output_path, "model.onnx")
            
            # Convert với tf2onnx
            spec = (tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="input"),)
            output_path_onnx, _ = tf2onnx.convert.from_function(
                model.signatures["serving_default"],
                input_signature=spec,
                opset=11,
                output_path=onnx_path
            )
            
            print(f"[ success ] TensorFlow model converted to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            print(f"[ error ] Failed to convert TensorFlow model to ONNX: {e}")
            return False
    
    @staticmethod
    def convert_sklearn_to_onnx(
        model,
        output_path: str,
        initial_types: List[Tuple[str, Any]],
        model_type: ModelType
    ) -> bool:
        """
        Convert scikit-learn model sang ONNX
        """
        try:
            from skl2onnx import convert_sklearn
            
            # Tạo output directory
            os.makedirs(output_path, exist_ok=True)
            onnx_path = os.path.join(output_path, "model.onnx")
            
            # Convert sklearn model
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_types,
                target_opset=11
            )
            
            # Lưu ONNX model
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"[ success ] Scikit-learn model converted to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            print(f"[ error ] Failed to convert scikit-learn model to ONNX: {e}")
            return False
    
    @staticmethod
    def _create_sample_inputs(
        model_type: ModelType, 
        tokenizer=None
    ) -> Dict[str, torch.Tensor]:
        """Tạo sample inputs cho các loại model khác nhau"""
        
        if model_type in [
            ModelType.TEXT_CLASSIFICATION,
            ModelType.TOKEN_CLASSIFICATION,
            ModelType.TRANSLATION,
            ModelType.SUMMARIZATION,
            ModelType.TEXT_GENERATION
        ]:
            if tokenizer:
                sample_text = "This is a sample text for model conversion."
                inputs = tokenizer(
                    sample_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                return {k: v for k, v in inputs.items()}
            else:
                return {
                    "input_ids": torch.randint(0, 1000, (1, 512)),
                    "attention_mask": torch.ones(1, 512)
                }
        
        elif model_type in [
            ModelType.IMAGE_CLASSIFICATION,
            ModelType.OBJECT_DETECTION,
            ModelType.IMAGE_TO_TEXT
        ]:
            return {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        elif model_type in [
            ModelType.TABULAR_CLASSIFICATION,
            ModelType.TABULAR_REGRESSION
        ]:
            return {"input": torch.randn(1, 10)}  # Giả định 10 features
        
        elif model_type in [
            ModelType.AUDIO_CLASSIFICATION,
            ModelType.AUTOMATIC_SPEECH_RECOGNITION
        ]:
            return {"input_values": torch.randn(1, 16000)}  # 1 second audio at 16kHz
        
        else:
            return {"input": torch.randn(1, 10)}
    
    @staticmethod
    def _save_preprocessing_config(
        output_path: str,
        model_type: ModelType,
        tokenizer=None
    ):
        """Lưu preprocessing config cho ONNX model"""
        
        config = {
            "model_type": model_type.value,
            "preprocessing": {}
        }
        
        if model_type in [
            ModelType.TEXT_CLASSIFICATION,
            ModelType.TOKEN_CLASSIFICATION,
            ModelType.TRANSLATION,
            ModelType.SUMMARIZATION,
            ModelType.TEXT_GENERATION
        ] and tokenizer:
            config["preprocessing"]["tokenizer"] = {
                "vocab_size": tokenizer.vocab_size,
                "max_length": getattr(tokenizer, "model_max_length", 512),
                "pad_token": tokenizer.pad_token,
                "unk_token": tokenizer.unk_token,
                "cls_token": getattr(tokenizer, "cls_token", None),
                "sep_token": getattr(tokenizer, "sep_token", None)
            }
        
        elif model_type in [
            ModelType.IMAGE_CLASSIFICATION,
            ModelType.OBJECT_DETECTION,
            ModelType.IMAGE_TO_TEXT
        ]:
            config["preprocessing"]["image"] = {
                "size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        
        # Lưu config file
        config_path = os.path.join(output_path, "preprocessing_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"[ success ] Preprocessing config saved: {config_path}")


class AutoConverter:
    """
    Auto converter để tự động detect và convert models
    """
    
    @staticmethod
    def detect_model_framework(model_path: str) -> Optional[ModelFramework]:
        """Tự động detect framework của model"""
        
        path = Path(model_path)
        
        # Check HuggingFace
        if (path / "config.json").exists() and (path / "pytorch_model.bin").exists():
            return ModelFramework.HUGGINGFACE
        
        # Check PyTorch
        if any(f.suffix == ".pt" or f.suffix == ".pth" for f in path.glob("*")):
            return ModelFramework.PYTORCH
        
        # Check TensorFlow
        if (path / "saved_model.pb").exists() or any(f.suffix == ".h5" for f in path.glob("*")):
            return ModelFramework.TENSORFLOW
        
        # Check ONNX
        if any(f.suffix == ".onnx" for f in path.glob("*")):
            return ModelFramework.ONNX
        
        # Check YOLO
        if any("yolo" in f.name.lower() and f.suffix == ".pt" for f in path.glob("*")):
            return ModelFramework.YOLO
        
        return None
    
    @staticmethod
    def auto_convert_to_onnx(
        model_path: str,
        output_path: str,
        model_type: ModelType,
        force_framework: Optional[ModelFramework] = None
    ) -> bool:
        """
        Tự động convert model sang ONNX dựa trên framework detect được
        """
        
        framework = force_framework or AutoConverter.detect_model_framework(model_path)
        
        if framework is None:
            print(f"[ error ] Cannot detect framework for model: {model_path}")
            return False
        
        print(f"[ info ] Detected framework: {framework.value}")
        
        if framework == ModelFramework.HUGGINGFACE:
            return ModelConverter.convert_huggingface_to_onnx(
                model_path, output_path, model_type
            )
        
        elif framework == ModelFramework.ONNX:
            print(f"[ info ] Model is already in ONNX format: {model_path}")
            return True
        
        else:
            print(f"[ warning ] Auto conversion not supported for framework: {framework.value}")
            return False 