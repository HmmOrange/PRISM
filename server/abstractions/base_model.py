from enum import Enum
from abc import ABC, abstractmethod

class ModelFramework(Enum):
    ULTRALYTIC = "ultralytic"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    TRANSFORMERS = "transformer"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    
class ModelType(Enum)
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question-answering"
    TEXT_GENERATION = "text-generation"
    SENTENCE_SIMILARITY = "sentence-similarity"
    SEQ2SEQ= "seq2seq"
    
    IMAGE_CLASSIFICATION = "image-classification"
    ZEROSHOT_IMAGE_CLASSIFICATION = "zeroshot-image-classification"
    OBJECT_DETECTION = "object-detection"
    ZEROSHOT_OBJECT_DETECTION = "zeroshot-object-detection"
    IMAGE_TO_TEXT = "image-to-text"
    IMAGE_TO_IMAGE = "image-to-image"
    
    AUDIO_CLASSIFICATION = "audio-classification"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    
    TABULAR_CLASSIFICATION = "tabular-classification"
    TABULAR_REGRESSION = "tabular-regression"
    
    VIDEO_CLASSIFICATION = "video-classification"
    
class BaseModelWrapper(ABC):
    def __init__(
        self,
        model_path: str,
        model_type: ModelType,
        framework: ModelFramework,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.framework = framework
        self.device = device
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self):
        raise NotImplementedError("load_model method must be implemented by subclasses")
    
    @abstractmethod
    def unload_model(self):
        raise NotImplementedError("unload_model method must be implemented by subclasses")
    
    @abstractmethod
    def predict(self, inputs):
        raise NotImplementedError("predict method must be implemented by subclasses")
    
    @abstractmethod
    def to_device(self, device: str):
        """
        Move the model to a specified device (e.g., 'cpu' or 'cuda').
        """
        raise NotImplementedError("to_device method must be implemented by subclasses")
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded in memory.
        """
        return self._is_loaded
    
    def get_model_info(self):
        pass
    
    def validate_inputs(self, inputs):
        pass
    
class ModelRegistry:
    _registry: Dict[ModelFramework, Dict[ModelType, type]] = {}
    
    @classmethod
    def register(cls, framework: ModelFramework, model_type: ModelType, model_class: type):
        if framework not in cls._registry:
            cls._registry[framework] = {}
        cls._registry[framework][model_type] = model_class
        