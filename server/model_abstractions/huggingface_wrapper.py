from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Any, Dict, List, Optional

from .base_model import BaseModelWrapper, ModelFramework, ModelType, ModelRegistry


class HuggingFaceWrapper(BaseModelWrapper):
    """
    HuggingFace Model Wrapper - hỗ trợ các model từ HuggingFace Hub
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
            framework=ModelFramework.HUGGINGFACE,
            device=device,
            metadata=metadata
        )
        self.pipeline = None
        self.kwargs = kwargs
    
    def load_model(self) -> None:
        """Load HuggingFace model"""
        try:
            if self.model_type == ModelType.SENTENCE_SIMILARITY:
                self.pipeline = SentenceTransformer(
                    self.model_path, 
                    trust_remote_code=True
                )
            elif self.model_type == ModelType.TOKEN_CLASSIFICATION:
                self.pipeline = hf_pipeline(
                    task=self.model_type.value,
                    model=self.model_path,
                    aggregation_strategy="simple",
                    trust_remote_code=True,
                    **self.kwargs
                )
            else:
                self.pipeline = hf_pipeline(
                    task=self.model_type.value,
                    model=self.model_path,
                    trust_remote_code=True,
                    **self.kwargs
                )
            
            self._is_loaded = True
            print(f"[ success ] HuggingFace model loaded: {self.model_path}")
        except Exception as e:
            print(f"[ error ] Failed to load HuggingFace model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload HuggingFace model"""
        if self.pipeline:
            self.pipeline = None
            self._is_loaded = False
            print(f"[ success ] HuggingFace model unloaded: {self.model_path}")
    
    def to_device(self, device: str) -> 'HuggingFaceWrapper':
        """Chuyển model sang device khác"""
        self.device = device
        
        if self._is_loaded and self.pipeline:
            try:
                if isinstance(self.pipeline, SentenceTransformer):
                    self.pipeline = self.pipeline.to(device)
                elif hasattr(self.pipeline, "model") and hasattr(self.pipeline.model, "to"):
                    self.pipeline.model.to(device)
                elif hasattr(self.pipeline, "to"):
                    self.pipeline.to(device)
                else:
                    print(f"[ info ] Skip device transfer for model: {self.model_path}")
            except Exception as e:
                print(f"[ warning ] Cannot move model to device: {e}")
        
        return self
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Thực hiện inference"""
        if not self._is_loaded:
            self.load_model()
        
        # Move to device if needed
        self.to_device(self.device)
        
        # Xử lý theo từng loại model
        if self.model_type == ModelType.TOKEN_CLASSIFICATION:
            text = inputs["text"]
            results = self.pipeline(text)
            return {
                "predicted": [
                    {
                        "word": item["word"],
                        "entity_group": item["entity_group"],
                    }
                    for item in results
                ]
            }
        
        elif self.model_type == ModelType.TEXT_CLASSIFICATION:
            text = inputs["text"]
            results = self.pipeline(text)
            return {"predicted": results[0]["label"]}
        
        elif self.model_type == ModelType.ZERO_SHOT_CLASSIFICATION:
            text = inputs["text"]
            labels = inputs["labels"]
            results = self.pipeline(text, candidate_labels=labels)
            return {"predicted": results["labels"][0]}
        
        elif self.model_type == ModelType.TRANSLATION:
            text = inputs["text"]
            results = self.pipeline(text)
            return {"translation_text": results[0]["translation_text"]}
        
        elif self.model_type == ModelType.SUMMARIZATION:
            text = inputs["text"]
            results = self.pipeline(text)
            return {"summary_text": results[0]["summary_text"]}
        
        elif self.model_type == ModelType.QUESTION_ANSWERING:
            context = inputs["context"]
            question = inputs["text"]
            results = self.pipeline(question=question, context=context)
            return {"answer": results["answer"]}
        
        elif self.model_type == ModelType.TEXT_GENERATION:
            text = inputs["text"]
            results = self.pipeline(text)
            return {"text": results[0]["generated_text"]}
        
        elif self.model_type == ModelType.SENTENCE_SIMILARITY:
            sentence = inputs["text"]
            other_sentences = inputs["other_sentences"]
            
            # Encode câu chính và các câu khác
            sentence_embedding = self.pipeline.encode([sentence])
            other_embeddings = self.pipeline.encode(other_sentences)
            
            similarities = cosine_similarity(sentence_embedding, other_embeddings)[0]
            scores = [float(similarity) for similarity in similarities]
            
            return {"predicted": scores}
        
        elif self.model_type == ModelType.IMAGE_CLASSIFICATION:
            image = inputs["image"]
            results = self.pipeline(image)
            predicted = results[0]["label"]
            
            # Map label nếu có id2label trong metadata
            if "id2label" in self.metadata:
                predicted = self.metadata["id2label"].get(predicted, predicted)
            
            return {"predicted": predicted}
        
        elif self.model_type == ModelType.IMAGE_TO_TEXT:
            image = inputs["image"]
            results = self.pipeline(image)
            return {"text": results[0]["generated_text"]}
        
        elif self.model_type == ModelType.AUTOMATIC_SPEECH_RECOGNITION:
            audio = inputs["audio"]
            results = self.pipeline(audio)
            return {"text": results["text"]}
        
        elif self.model_type == ModelType.AUDIO_CLASSIFICATION:
            audio = inputs["audio"]
            results = self.pipeline(audio)
            return {"predicted": results[0]["label"]}
        
        elif self.model_type == ModelType.VIDEO_CLASSIFICATION:
            video = inputs["video"].replace("\\", "/")
            results = self.pipeline(video)
            return {"predicted": results[0]["label"]}
        
        else:
            # Fallback cho các task khác
            try:
                results = self.pipeline(inputs)
                return {"output": results}
            except Exception as e:
                raise ValueError(f"Unsupported model type or invalid inputs: {e}")


# Đăng ký HuggingFace wrappers
def register_huggingface_wrappers():
    """Đăng ký tất cả HuggingFace wrappers vào registry"""
    
    # Text tasks
    text_tasks = [
        ModelType.TEXT_CLASSIFICATION,
        ModelType.TOKEN_CLASSIFICATION,
        ModelType.ZERO_SHOT_CLASSIFICATION,
        ModelType.TRANSLATION,
        ModelType.SUMMARIZATION,
        ModelType.QUESTION_ANSWERING,
        ModelType.TEXT_GENERATION,
        ModelType.SENTENCE_SIMILARITY
    ]
    
    for model_type in text_tasks:
        ModelRegistry.register(ModelFramework.HUGGINGFACE, model_type, HuggingFaceWrapper)
    
    # Image tasks
    image_tasks = [
        ModelType.IMAGE_CLASSIFICATION,
        ModelType.IMAGE_TO_TEXT
    ]
    
    for model_type in image_tasks:
        ModelRegistry.register(ModelFramework.HUGGINGFACE, model_type, HuggingFaceWrapper)
    
    # Audio tasks
    audio_tasks = [
        ModelType.AUDIO_CLASSIFICATION,
        ModelType.AUTOMATIC_SPEECH_RECOGNITION
    ]
    
    for model_type in audio_tasks:
        ModelRegistry.register(ModelFramework.HUGGINGFACE, model_type, HuggingFaceWrapper)
    
    # Video tasks
    ModelRegistry.register(ModelFramework.HUGGINGFACE, ModelType.VIDEO_CLASSIFICATION, HuggingFaceWrapper)


# Auto register khi import
register_huggingface_wrappers() 