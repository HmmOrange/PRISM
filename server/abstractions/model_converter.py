from server.abstractions.base_model import ModelType
import pytorch
from optimum

class ModelConverter:
    @staticmethod
    def convert_transformers_to_onnx(
        self, model_path: str, save_path: str, model_type: ModelType, opset_version: int = 11
    ):
        pass
    
    @staticmethod
    def convert_ultralytics_to_onnx(
        self
    ):
        pass
    
    @staticmethod
    def convert_tensorflow_to_onnx(
        self
    ):
        pass

    
    @staticmethod
    def convert_keras_to_onnx(
        self
    ):
        pass

    @staticmethod
    def convert_sentence_transformers_to_onnx(
        self
    ):
        pass
    
    @staticmethod
    def auto_convert_to_onnx(
        self
    ):
        pass