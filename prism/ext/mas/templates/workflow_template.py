workflow_template = """from scripts.model_inference import model_inference
from typing import Union

class Workflow:
    async def __call__(self, data: dict) -> Union[str, int, float]:
        '''
        Process a single testcase and return prediction
        
        Args:
            data: {{
                "image_paths": [list of image file paths],
                "video_paths": [list of video file paths], 
                "audio_paths": [list of audio file paths],
                "text_data": {{dict of text data}}
            }}
            
        Returns:
            prediction: prediction for this testcase
        '''
"""