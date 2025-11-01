import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoConfig, AutoImageProcessor, AutoModel
import gc
from pathlib import Path

from server.settings import ROOT_PATH
from server.abstractions.converter.base_converter import BaseConverter

class TransformersConverter(BaseConverter):
    def convert_to_onnx(self, model_path: str, output_path: str):
        model = AutoModel.from_pretrained(model_path)
        # Dùng AutoImageProcessor cho model xử lý ảnh như ViT
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        
        feature = "image-classification"  # đúng loại tác vụ với ViT
        
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        onnx_config = model_onnx_config(model.config)
        
        transformers.onnx.export(
            preprocessor=image_processor,
            model=model,
            config=onnx_config,
            opset=14,
            output=Path(output_path)  # dùng tham số đầu vào
        )
        
        del model, image_processor
        gc.collect()

def run()

if __name__ == '__main__':
    path = f"{ROOT_PATH}/server/models/semihdervis/cat-emotion-classifier"
    output = f"{ROOT_PATH}/server/models/semihdervis/cat-emotion-classifier.onnx"
    
    converter = TransformersConverter()
    converter.convert_to_onnx(path, output)

