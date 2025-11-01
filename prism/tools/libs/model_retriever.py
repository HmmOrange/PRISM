from prism.tools.tool_registry import register_tool
from prism.llm import LLM


tags = ["model retriever", "model information"]

@register_tool(tags=tags)
class ModelRetriever:
    """
    Retrieve model information from the model hub.
    Relating to Machine Leraning Problem.
    """

    def retrieve(self, model_description: str, model_type: str) -> str:
        """
        Retrieve model information from the model hub.
        
        Args:
            model_description (str): The description about the model which you want to use. More detail more accuracy.
            model_type (str): The type of the model you want to use. It should belong to ["token-classification", "text-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]
            
        Returns:
            str: The model information.
        
        """
        llm = LLM()
        return ""
