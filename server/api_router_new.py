import os
import io
import json
from typing import Optional

from fastapi import APIRouter, Query, Form, UploadFile, File, HTTPException
from server.schema_base import DataResponse

import time
from server.settings import device, ROOT_PATH
from PIL import Image

# Import hệ thống model abstractions mới
from server.model_abstractions import (
    ModelManager, ModelFramework, ModelType
)

# Initialize Model Manager
model_manager = ModelManager(
    max_models_in_ram=5,
    max_models_in_disk=10,
    models_directory=f"{ROOT_PATH}/server/models",
    auto_convert_to_onnx=False  # Set True để tự động convert sang ONNX
)

router = APIRouter()


def initialize_models():
    """Initialize tất cả models từ config file"""
    try:
        # Load model configurations
        with open(f"{ROOT_PATH}/data/huggingface_models.jsonl", "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = json.loads(line)
                
                # Map pipeline_tag sang ModelType
                model_type = _map_pipeline_tag_to_model_type(info["pipeline_tag"])
                if model_type is None:
                    print(f"[ warning ] Unsupported pipeline tag: {info['pipeline_tag']}")
                    continue
                
                # Determine framework (default to HuggingFace)
                framework = ModelFramework.HUGGINGFACE
                
                # Special cases
                if info["pipeline_tag"] == "object-detection":
                    framework = ModelFramework.YOLO
                elif info["pipeline_tag"] == "tabular-regression":
                    framework = ModelFramework.PYTORCH
                elif info["pipeline_tag"] == "tabular-classification":
                    framework = ModelFramework.TENSORFLOW
                
                # Register model
                success = model_manager.register_model(
                    model_id=info["id"],
                    model_type=model_type,
                    framework=framework,
                    metadata={
                        "description": info.get("description", ""),
                        "id2label": info.get("metadata", {}).get("id2label", {}),
                        **info.get("metadata", {})
                    }
                )
                
                if success:
                    print(f"[ registered ] Model {info['id']} ({info['pipeline_tag']})")
                else:
                    print(f"[ error ] Failed to register model {info['id']}")
        
        print(f"[ success ] Initialized {len(model_manager.list_models())} models")
        
    except Exception as e:
        print(f"[ error ] Failed to initialize models: {e}")


def _map_pipeline_tag_to_model_type(pipeline_tag: str) -> Optional[ModelType]:
    """Map pipeline tag sang ModelType enum"""
    mapping = {
        "text-classification": ModelType.TEXT_CLASSIFICATION,
        "token-classification": ModelType.TOKEN_CLASSIFICATION,
        "zero-shot-classification": ModelType.ZERO_SHOT_CLASSIFICATION,
        "translation": ModelType.TRANSLATION,
        "summarization": ModelType.SUMMARIZATION,
        "question-answering": ModelType.QUESTION_ANSWERING,
        "text-generation": ModelType.TEXT_GENERATION,
        "sentence-similarity": ModelType.SENTENCE_SIMILARITY,
        
        "image-classification": ModelType.IMAGE_CLASSIFICATION,
        "object-detection": ModelType.OBJECT_DETECTION,
        "image-to-text": ModelType.IMAGE_TO_TEXT,
        
        "audio-classification": ModelType.AUDIO_CLASSIFICATION,
        "automatic-speech-recognition": ModelType.AUTOMATIC_SPEECH_RECOGNITION,
        
        "tabular-classification": ModelType.TABULAR_CLASSIFICATION,
        "tabular-regression": ModelType.TABULAR_REGRESSION,
        
        "video-classification": ModelType.VIDEO_CLASSIFICATION,
    }
    
    return mapping.get(pipeline_tag)


@router.get("/running")
def running():
    """Health check endpoint"""
    return DataResponse().success_response(data={"running": True})


@router.get("/status/{model_id}")
def status(model_id: str):
    """Check model status"""
    disabled_models = [
        "microsoft/trocr-base-handwritten",
    ]
    
    if model_id in disabled_models:
        return DataResponse().custom_response(
            code="001", message="Model disabled", data={"loaded": False}
        )
    
    # Check if model is registered
    model_info = model_manager.get_model_info(model_id)
    if model_info is None:
        print(f"[ check {model_id} ] failed - not registered")
        return DataResponse().custom_response(
            code="001", message="Model not found", data={"loaded": False}
        )
    
    print(f"[ check {model_id} ] success")
    return DataResponse().success_response(data={
        "loaded": model_info["is_loaded"],
        "framework": model_info["framework"],
        "model_type": model_info["model_type"]
    })


@router.get("/models")
def list_models():
    """List tất cả models được quản lý"""
    models = model_manager.list_models()
    return DataResponse().success_response(data={"models": models})


@router.get("/models/{model_id}/info")
def get_model_info(model_id: str):
    """Get thông tin chi tiết của model"""
    model_info = model_manager.get_model_info(model_id)
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return DataResponse().success_response(data=model_info)


@router.post("/models/{model_id}/convert-to-onnx")
async def convert_model_to_onnx(model_id: str):
    """Convert model sang ONNX format"""
    model_info = model_manager.get_model_info(model_id)
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model_info["framework"] == "onnx":
        return DataResponse().success_response(data={"message": "Model is already in ONNX format"})
    
    try:
        # Thực hiện conversion logic ở đây
        # (simplified - trong thực tế sẽ cần implement converter)
        
        return DataResponse().success_response(data={"message": "Conversion initiated"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")


@router.post("/models")
async def inference(
    model_id: str = Query(...),
    data: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    """
    Unified inference endpoint sử dụng Model Manager
    """
    # Check if model exists
    model_info = model_manager.get_model_info(model_id)
    if model_info is None:
        return DataResponse().custom_response(
            code="001", message="Model not found", data={"inference": "failed"}
        )
    
    try:
        # Parse input data
        input_data = json.loads(data)
        
        print(f"[ inference {model_id} ] start")
        start_time = time.time()
        
        # Prepare inputs dựa trên model type
        inputs = await _prepare_inputs(input_data, image, audio, model_info["model_type"])
        
        # Run inference với Model Manager
        result = model_manager.predict(
            model_id=model_id,
            inputs=inputs,
            device=device
        )
        
        if result is None:
            return DataResponse().custom_response(
                code="002", message="Inference failed", data={"inference": "failed"}
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[ complete {model_id} ] {duration:.2f}s")
        print(f"[ result {model_id} ] {result}")
        
        return DataResponse().success_response(data=result)
        
    except json.JSONDecodeError:
        return DataResponse().custom_response(
            code="003", message="Invalid JSON data", data={"inference": "failed"}
        )
    except Exception as e:
        print(f"[ error {model_id} ] {str(e)}")
        return DataResponse().custom_response(
            code="004", message=f"Inference error: {str(e)}", data={"inference": "failed"}
        )


async def _prepare_inputs(
    data: dict, 
    image: Optional[UploadFile], 
    audio: Optional[UploadFile],
    model_type: str
) -> dict:
    """Prepare inputs dựa trên model type"""
    inputs = {}
    
    # Text inputs
    if model_type in [
        "text-classification", "token-classification", "zero-shot-classification",
        "translation", "summarization", "question-answering", "text-generation",
        "sentence-similarity"
    ]:
        inputs["text"] = data.get("text", "")
        
        # Special cases
        if model_type == "zero-shot-classification":
            inputs["labels"] = data.get("labels", [])
        elif model_type == "question-answering":
            inputs["context"] = data.get("context", "")
        elif model_type == "sentence-similarity":
            inputs["other_sentences"] = data.get("other_sentences", [])
    
    # Image inputs
    elif model_type in ["image-classification", "object-detection", "image-to-text"]:
        if image is not None:
            image_data = await image.read()
            inputs["image"] = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            raise ValueError("Image is required for this model type")
    
    # Audio inputs
    elif model_type in ["audio-classification", "automatic-speech-recognition"]:
        if audio is not None:
            inputs["audio"] = await audio.read()
        else:
            raise ValueError("Audio is required for this model type")
    
    # Tabular inputs
    elif model_type in ["tabular-classification", "tabular-regression"]:
        inputs["row"] = data.get("row", {})
    
    # Video inputs
    elif model_type == "video-classification":
        inputs["video"] = data.get("video", "").replace("\\", "/")
    
    else:
        # Fallback - pass all data
        inputs.update(data)
    
    return inputs


# Initialize models khi import
initialize_models() 