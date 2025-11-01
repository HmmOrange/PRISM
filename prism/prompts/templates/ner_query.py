from prism.schema.message import Message

ner_query_system = """Your task is to extract model-related information from user queries about machine learning models.

You MUST respond with a JSON object that contains exactly these fields (no more, no less):
- "model_type": Array of model architectures mentioned (e.g., ["vit", "bert", "gpt2", "resnet", "t5", "roberta"])
- "task_type": Array of ML tasks from: ["token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]
- "datasets": Array of datasets mentioned (e.g., ["imagenet-1k", "cifar100", "wikitext", "cc100", "squad"])
- "resolution": Array of input resolutions mentioned (e.g., ["224x224", "16khz", "22050hz"])
- "classes": Array of classification labels mentioned (e.g., ["airplane", "cat", "fake news", "real news"])
- "language": Array of languages mentioned (e.g., ["english", "vietnamese", "multilingual", "en", "vi"])
- "domain": Array of application domains mentioned (e.g., ["vision", "nlp", "speech", "audio", "video", "tabular"])
- "num_classes": Array of number of classes mentioned (e.g., ["2", "1000", "21843"])
- "tags": Array of model tags/labels mentioned (e.g., ["pytorch", "vision", "transformers", "text-classification"])
- "input_format": Array of input format requirements mentioned (e.g., ["vi:", "en:", "text less than 500 words", "max_length 512"])
- "output_format": Array of output format details mentioned (e.g., ["LABEL_0", "LABEL_1", "class_names", "probabilities", "logits"])

SPECIAL REQUIREMENTS FOR CLASSIFICATION TASKS:
For classification tasks (image-classification, text-classification, audio-classification, video-classification, tabular-classification), you should STRONGLY INFER and provide:
1. "datasets": MANDATORY - Infer the most appropriate dataset based on context. For example:
   - Image classification → "imagenet-1k", "cifar10", "cifar100", "mnist"
   - Text classification → "imdb", "sst2", "ag_news", "yelp_reviews"
   - Audio classification → "speech_commands", "urban_sound", "gtzan"
   - Video classification → "kinetics", "ucf101", "something_something"
2. "num_classes": Try to infer number of classes from context or provide common defaults
3. "classes": If specific classes are mentioned or can be inferred from domain

Extract information that the user is looking for or interested in. If information for any field is not found, use an empty array [] EXCEPT for classification tasks where you should make reasonable inferences for datasets.
Focus on what the user wants to find or use, not what they already have.
"""

query_prompt_one_shot_input = """Use image-classification to classify the image into one of the 1000 ImageNet categories"""

query_prompt_one_shot_output = """
```json
{
    "model_type": [],
    "task_type": ["image-classification"],
    "datasets": ["imagenet-1k"],
    "resolution": [],
    "classes": [],
    "language": [],
    "domain": ["vision"],
    "num_classes": ["1000"],
    "tags": ["image-classification"],
    "input_format": [],
    "output_format": []
}
```
"""

query_prompt_two_shot_input = """I need a text classifier for sentiment analysis"""

query_prompt_two_shot_output = """
```json
{
    "model_type": [],
    "task_type": ["text-classification"],
    "datasets": ["imdb", "sst2"],
    "resolution": [],
    "classes": ["positive", "negative"],
    "language": [],
    "domain": ["nlp"],
    "num_classes": ["2"],
    "tags": ["text-classification", "sentiment-analysis"],
    "input_format": [],
    "output_format": []
}
```
"""

query_prompt_three_shot_input = """Audio classification model to recognize 10 different urban sounds"""

query_prompt_three_shot_output = """
```json
{
    "model_type": [],
    "task_type": ["audio-classification"],
    "datasets": ["urban_sound", "esc-10"],
    "resolution": [],
    "classes": [],
    "language": [],
    "domain": ["audio"],
    "num_classes": ["10"],
    "tags": ["audio-classification"],
    "input_format": [],
    "output_format": []
}
```
"""

prompt_template = [
    Message(role="system", content=ner_query_system),
    Message(role="user", content=query_prompt_one_shot_input),
    Message(role="assistant", content=query_prompt_one_shot_output),
    Message(role="user", content=query_prompt_two_shot_input),
    Message(role="assistant", content=query_prompt_two_shot_output),
    Message(role="user", content=query_prompt_three_shot_input),
    Message(role="assistant", content=query_prompt_three_shot_output),
    Message(role="user", content="${query}")
]