from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
from typing import Any, Dict, List, Optional

from .base_model import BaseModelWrapper, ModelFramework, ModelType, ModelRegistry


class YOLOWrapper(BaseModelWrapper):
    """
    YOLO Model Wrapper - hỗ trợ YOLO models từ Ultralytics
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            model_type=ModelType.OBJECT_DETECTION,
            framework=ModelFramework.YOLO,
            device=device,
            metadata=metadata
        )
        self.yolo_model = None
    
    def load_model(self) -> None:
        """Load YOLO model"""
        try:
            # YOLO model path thường là .pt file
            yolo_model_path = f"{self.model_path}/yolo11l.pt"
            self.yolo_model = YOLO(yolo_model_path)
            self._is_loaded = True
            print(f"[ success ] YOLO model loaded: {self.model_path}")
        except Exception as e:
            print(f"[ error ] Failed to load YOLO model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload YOLO model"""
        if self.yolo_model:
            self.yolo_model = None
            self._is_loaded = False
            print(f"[ success ] YOLO model unloaded: {self.model_path}")
    
    def to_device(self, device: str) -> 'YOLOWrapper':
        """Chuyển YOLO model sang device khác"""
        self.device = device
        
        if self._is_loaded and self.yolo_model:
            try:
                # YOLO có thể tự động handle device
                self.yolo_model.to(device)
            except Exception as e:
                print(f"[ warning ] Cannot move YOLO model to device: {e}")
        
        return self
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Thực hiện inference với YOLO"""
        if not self._is_loaded:
            self.load_model()
        
        # Lấy image data
        image_data = inputs.get("image")
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image = image_data
        
        # Run YOLO inference
        results = self.yolo_model(image)
        
        predictions = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls in zip(boxes, scores, classes):
                label = self.yolo_model.names[cls]
                predictions.append({
                    "label": label,
                    "score": float(score),
                    "box": {
                        "xmin": int(box[0]),
                        "ymin": int(box[1]),
                        "xmax": int(box[2]),
                        "ymax": int(box[3]),
                    },
                })
        
        return {"predicted": predictions}


class TabularWrapper(BaseModelWrapper):
    """
    Wrapper cho các custom tabular pipelines
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: ModelType,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        feature_order: Optional[List[str]] = None,
        label_map: Optional[Dict[int, str]] = None,
        **kwargs
    ):
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            framework=ModelFramework.PYTORCH if model_type == ModelType.TABULAR_REGRESSION else ModelFramework.TENSORFLOW,
            device=device,
            metadata=metadata
        )
        self.feature_order = feature_order
        self.label_map = label_map
        self.custom_pipeline = None
    
    def load_model(self) -> None:
        """Load custom tabular pipeline"""
        try:
            if self.model_type == ModelType.TABULAR_REGRESSION:
                from server.custom_pipeline.tabular_pipeline import TabularRegressionPipeline
                self.custom_pipeline = TabularRegressionPipeline(
                    model_path=self.model_path,
                    feature_order=self.feature_order or [
                        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                        "Population", "AveOccup", "Latitude", "Longitude"
                    ]
                )
            elif self.model_type == ModelType.TABULAR_CLASSIFICATION:
                from server.custom_pipeline.tabular_pipeline import TabularClassificationPipeline
                self.custom_pipeline = TabularClassificationPipeline(
                    model_path=self.model_path,
                    feature_order=self.feature_order or [
                        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
                        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
                        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
                        "V28", "Amount"
                    ],
                    label_map=self.label_map or {0: "0", 1: "1"}
                )
            else:
                raise ValueError(f"Unsupported tabular model type: {self.model_type}")
            
            self._is_loaded = True
            print(f"[ success ] Tabular model loaded: {self.model_path}")
        except Exception as e:
            print(f"[ error ] Failed to load tabular model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload tabular model"""
        if self.custom_pipeline:
            self.custom_pipeline = None
            self._is_loaded = False
            print(f"[ success ] Tabular model unloaded: {self.model_path}")
    
    def to_device(self, device: str) -> 'TabularWrapper':
        """Chuyển tabular model sang device khác"""
        self.device = device
        
        if self._is_loaded and self.custom_pipeline:
            try:
                self.custom_pipeline.to(device)
            except Exception as e:
                print(f"[ warning ] Cannot move tabular model to device: {e}")
        
        return self
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Thực hiện inference với tabular model"""
        if not self._is_loaded:
            self.load_model()
        
        row = inputs.get("row", {})
        results = self.custom_pipeline(row)
        
        if self.model_type == ModelType.TABULAR_REGRESSION:
            return {"predicted": results[0]["prediction"]}
        else:
            return {"predicted": results[0]["label"]}


# Đăng ký specialized wrappers
def register_specialized_wrappers():
    """Đăng ký các specialized wrappers vào registry"""
    
    # YOLO cho object detection
    ModelRegistry.register(ModelFramework.YOLO, ModelType.OBJECT_DETECTION, YOLOWrapper)
    
    # Custom tabular pipelines
    ModelRegistry.register(ModelFramework.PYTORCH, ModelType.TABULAR_REGRESSION, TabularWrapper)
    ModelRegistry.register(ModelFramework.TENSORFLOW, ModelType.TABULAR_CLASSIFICATION, TabularWrapper)


# Auto register khi import
register_specialized_wrappers() 