import onnxruntime as ort
import numpy as np
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

from .base_model import BaseModelWrapper, ModelFramework, ModelType, ModelRegistry


class ONNXModelWrapper(BaseModelWrapper):
    """
    ONNX Model Wrapper - hỗ trợ các model đã được convert sang ONNX format
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: ModelType,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            framework=ModelFramework.ONNX,
            device=device,
            metadata=metadata
        )
        self.session = None
        self.providers = self._get_providers(device)
        
        # Lấy preprocessing config từ metadata nếu có
        self.preprocessing_config = self.metadata.get("preprocessing", {})
        
    def _get_providers(self, device: str) -> List[str]:
        """Xác định ONNX execution providers dựa trên device"""
        if device.startswith("cuda"):
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == "cpu":
            return ['CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def load_model(self) -> None:
        """Load ONNX model"""
        try:
            onnx_model_path = f"{self.model_path}/model.onnx"
            self.session = ort.InferenceSession(
                onnx_model_path, 
                providers=self.providers
            )
            self._is_loaded = True
            print(f"[ success ] ONNX model loaded: {self.model_path}")
        except Exception as e:
            print(f"[ error ] Failed to load ONNX model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload ONNX model"""
        if self.session:
            self.session = None
            self._is_loaded = False
            print(f"[ success ] ONNX model unloaded: {self.model_path}")
    
    def to_device(self, device: str) -> 'ONNXModelWrapper':
        """Chuyển model sang device khác bằng cách thay đổi providers"""
        self.device = device
        self.providers = self._get_providers(device)
        
        # Reload model với providers mới
        if self._is_loaded:
            self.unload_model()
            self.load_model()
        
        return self
    
    def _preprocess_text(self, text: str) -> Dict[str, np.ndarray]:
        """Preprocessing cho text input"""
        # Đây là ví dụ đơn giản - trong thực tế cần tokenizer
        # Có thể load tokenizer config từ metadata
        if "tokenizer" in self.preprocessing_config:
            # TODO: Implement proper tokenization
            pass
        
        # Placeholder cho text preprocessing
        # Trong thực tế sẽ cần encode text thành token ids
        return {"input_ids": np.array([[0, 1, 2]])}  # Dummy
    
    def _preprocess_image(self, image_data: Any) -> Dict[str, np.ndarray]:
        """Preprocessing cho image input"""
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            raise ValueError("Unsupported image format")
        
        # Default image preprocessing
        size = self.preprocessing_config.get("size", (224, 224))
        mean = self.preprocessing_config.get("mean", [0.485, 0.456, 0.406])
        std = self.preprocessing_config.get("std", [0.229, 0.224, 0.225])
        
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        tensor = transform(image).unsqueeze(0)
        return {"pixel_values": tensor.numpy()}
    
    def _preprocess_tabular(self, row: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Preprocessing cho tabular data"""
        feature_order = self.preprocessing_config.get("feature_order", list(row.keys()))
        
        # Sắp xếp features theo thứ tự đã định
        features = [row[key] for key in feature_order if key in row]
        
        # Normalization nếu có config
        if "normalization" in self.preprocessing_config:
            norm_config = self.preprocessing_config["normalization"]
            means = norm_config.get("means", [])
            stds = norm_config.get("stds", [])
            
            if means and stds:
                features = [(f - m) / s for f, m, s in zip(features, means, stds)]
        
        return {"input": np.array([features], dtype=np.float32)}
    
    def _postprocess_classification(self, outputs: np.ndarray) -> Dict[str, Any]:
        """Postprocessing cho classification tasks"""
        if len(outputs.shape) == 1:
            predicted_class = int(np.argmax(outputs))
            confidence = float(outputs[predicted_class])
        else:
            predicted_class = int(np.argmax(outputs[0]))
            confidence = float(outputs[0][predicted_class])
        
        # Map class index sang label nếu có
        label_map = self.metadata.get("label_map", {})
        if label_map:
            label = label_map.get(str(predicted_class), str(predicted_class))
        else:
            label = str(predicted_class)
        
        return {
            "predicted": label,
            "confidence": confidence,
            "class_id": predicted_class
        }
    
    def _postprocess_regression(self, outputs: np.ndarray) -> Dict[str, Any]:
        """Postprocessing cho regression tasks"""
        if len(outputs.shape) == 1:
            prediction = float(outputs[0])
        else:
            prediction = float(outputs[0][0])
        
        return {"predicted": prediction}
    
    def _postprocess_object_detection(self, outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Postprocessing cho object detection"""
        # Giả định outputs có boxes, scores, classes
        boxes = outputs.get("boxes", np.array([]))
        scores = outputs.get("scores", np.array([]))
        classes = outputs.get("classes", np.array([]))
        
        detections = []
        confidence_threshold = self.preprocessing_config.get("confidence_threshold", 0.5)
        
        for box, score, cls in zip(boxes, scores, classes):
            if score > confidence_threshold:
                detections.append({
                    "box": {
                        "xmin": float(box[0]),
                        "ymin": float(box[1]),
                        "xmax": float(box[2]),
                        "ymax": float(box[3])
                    },
                    "score": float(score),
                    "label": str(int(cls))
                })
        
        return {"predicted": detections}
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Thực hiện inference"""
        if not self._is_loaded:
            self.load_model()
        
        # Preprocessing dựa trên model type
        if self.model_type in [
            ModelType.TEXT_CLASSIFICATION,
            ModelType.TOKEN_CLASSIFICATION,
            ModelType.ZERO_SHOT_CLASSIFICATION,
            ModelType.TRANSLATION,
            ModelType.SUMMARIZATION,
            ModelType.QUESTION_ANSWERING,
            ModelType.TEXT_GENERATION
        ]:
            processed_inputs = self._preprocess_text(inputs.get("text", ""))
        
        elif self.model_type in [
            ModelType.IMAGE_CLASSIFICATION,
            ModelType.OBJECT_DETECTION,
            ModelType.IMAGE_TO_TEXT
        ]:
            processed_inputs = self._preprocess_image(inputs.get("image"))
        
        elif self.model_type in [
            ModelType.TABULAR_CLASSIFICATION,
            ModelType.TABULAR_REGRESSION
        ]:
            processed_inputs = self._preprocess_tabular(inputs.get("row", {}))
        
        else:
            # Fallback - sử dụng inputs trực tiếp
            processed_inputs = inputs
        
        # Run inference
        input_names = [input.name for input in self.session.get_inputs()]
        output_names = [output.name for output in self.session.get_outputs()]
        
        # Map processed inputs to ONNX input names
        onnx_inputs = {}
        for name in input_names:
            if name in processed_inputs:
                onnx_inputs[name] = processed_inputs[name]
            elif len(processed_inputs) == 1 and len(input_names) == 1:
                # Nếu chỉ có 1 input, map trực tiếp
                onnx_inputs[name] = list(processed_inputs.values())[0]
        
        outputs = self.session.run(output_names, onnx_inputs)
        
        # Postprocessing dựa trên model type
        if len(outputs) == 1:
            output_array = outputs[0]
        else:
            # Multiple outputs - tạo dict
            output_dict = {name: output for name, output in zip(output_names, outputs)}
            return self._postprocess_multiple_outputs(output_dict)
        
        # Single output postprocessing
        if self.model_type in [
            ModelType.TEXT_CLASSIFICATION,
            ModelType.IMAGE_CLASSIFICATION,
            ModelType.TABULAR_CLASSIFICATION
        ]:
            return self._postprocess_classification(output_array)
        
        elif self.model_type in [
            ModelType.TABULAR_REGRESSION
        ]:
            return self._postprocess_regression(output_array)
        
        elif self.model_type == ModelType.OBJECT_DETECTION:
            # Cho object detection, outputs thường có nhiều tensor
            return self._postprocess_object_detection({output_names[0]: output_array})
        
        else:
            # Default - trả về raw output
            return {"output": output_array.tolist()}
    
    def _postprocess_multiple_outputs(self, outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Xử lý multiple outputs"""
        if self.model_type == ModelType.OBJECT_DETECTION:
            return self._postprocess_object_detection(outputs)
        
        # Default cho multiple outputs
        result = {}
        for name, output in outputs.items():
            result[name] = output.tolist()
        return result


# Đăng ký các ONNX wrappers
def register_onnx_wrappers():
    """Đăng ký tất cả ONNX wrappers vào registry"""
    
    # Text tasks
    for model_type in [
        ModelType.TEXT_CLASSIFICATION,
        ModelType.TOKEN_CLASSIFICATION,
        ModelType.ZERO_SHOT_CLASSIFICATION,
        ModelType.TRANSLATION,
        ModelType.SUMMARIZATION,
        ModelType.QUESTION_ANSWERING,
        ModelType.TEXT_GENERATION,
        ModelType.SENTENCE_SIMILARITY
    ]:
        ModelRegistry.register(ModelFramework.ONNX, model_type, ONNXModelWrapper)
    
    # Image tasks
    for model_type in [
        ModelType.IMAGE_CLASSIFICATION,
        ModelType.OBJECT_DETECTION,
        ModelType.IMAGE_TO_TEXT
    ]:
        ModelRegistry.register(ModelFramework.ONNX, model_type, ONNXModelWrapper)
    
    # Audio tasks
    for model_type in [
        ModelType.AUDIO_CLASSIFICATION,
        ModelType.AUTOMATIC_SPEECH_RECOGNITION
    ]:
        ModelRegistry.register(ModelFramework.ONNX, model_type, ONNXModelWrapper)
    
    # Tabular tasks
    for model_type in [
        ModelType.TABULAR_CLASSIFICATION,
        ModelType.TABULAR_REGRESSION
    ]:
        ModelRegistry.register(ModelFramework.ONNX, model_type, ONNXModelWrapper)
    
    # Video tasks
    ModelRegistry.register(ModelFramework.ONNX, ModelType.VIDEO_CLASSIFICATION, ONNXModelWrapper)


# Auto register khi import
register_onnx_wrappers() 