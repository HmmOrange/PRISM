from abc import ABC, abstractmethod

class BaseConverter(ABC):
    @abstractmethod
    def convert_to_onnx(self, model_path: str, output_path: str):
        pass
    