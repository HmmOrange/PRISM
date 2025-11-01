from prism.schema.message import Message


system = f"""You are an expert at analyzing user requirements and providing detailed model descriptions for specified model types.

## Your Task:
Analyze the user requirement and provide detailed model description for the specified model type.

## For model_description, include these focused details:
- **Key Datasets**: Most relevant datasets for this model type (2-3 main ones)
- **Output Labels**: Specific label types and categories relevant to the task
- **Primary Use Cases**: Main applications (2-3 focused examples)
- **Technical Keywords**: Essential technical terms

You must respond with the following XML structure:
<thought>
[Analyze how to solve the user requirement with the specified model type. Think about the approach and how the model type fits the task.]
</thought>
<model_description>
[Focused and specific keywords for the specified model type. Include key datasets, relevant labels, main use cases, and technical terms - keep it concise but comprehensive.]
</model_description>
<model_type>
str: [Which model type used in the user requirement. Choose one in ["token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]]
</model_type>"""

# Example 1: Image Classification
one_shot_user_requirement_1 = """Use image-classification to automatically categorize product images in an e-commerce platform. The system should identify different types of products like clothing, electronics, furniture, etc. from uploaded product photos to organize them into the correct categories."""

one_shot_output_1 = """<thought>
For this e-commerce product categorization task using image-classification, the model needs to understand various product types across different domains. Image classification models take a full image as input and output a single category prediction. The approach would be to use models trained on diverse datasets that include product images or general object recognition. The challenge is ensuring accurate distinction between visually similar products from different categories.
</thought>
<model_description>
**Key Datasets**: ImageNet-1k (general object recognition), Fashion-MNIST (clothing items), Product-10K (e-commerce products)

**Output Labels**: Product categories: clothing, electronics, furniture, home goods, accessories; specific items: shirts, smartphones, chairs, kitchenware

**Primary Use Cases**: E-commerce product categorization, automated inventory management, visual search and recommendation

**Technical Keywords**: product classification, convolutional neural networks, transfer learning, multi-class classification, computer vision, retail automation
</model_description>
<model_type>
image-classification
</model_type>"""

# Example 2: Object Detection  
one_shot_user_requirement_2 = """Use object-detection to detect and locate multiple objects in retail store images for inventory management. The system should identify products on shelves, count them, and determine their positions to track stock levels and optimize shelf layouts."""

one_shot_output_2 = """<thought>
For this retail inventory management task using object-detection, the model needs to detect and locate multiple products simultaneously in store images. Object detection models identify objects and draw bounding boxes around them with class predictions. The approach would involve using models trained on retail-specific datasets or general object detection datasets. Key challenges include handling small objects, products partially obscured by others, and maintaining accuracy across various store layouts.
</thought>
<model_description>
**Key Datasets**: COCO-2017 (general object detection), SKU-110K (dense retail products), RPC dataset (retail checkout)

**Output Labels**: Bounding boxes with coordinates, confidence scores, retail products: beverages, snacks, household items, personal care, packaged goods

**Primary Use Cases**: Retail inventory management, automated stock counting, shelf compliance monitoring

**Technical Keywords**: object detection, bounding boxes, multi-object recognition, YOLO, R-CNN, real-time detection, inventory tracking
</model_description>
<model_type>
object-detection
</model_type>"""

usr_input = """${user_requirement}"""

# Use image-classification example by default, but both examples are available
prompt_template = [
    Message(role="system", content=system),
    Message(role="user", content=one_shot_user_requirement_1),
    Message(role="assistant", content=one_shot_output_1),
    Message(role="user", content=one_shot_user_requirement_2), 
    Message(role="assistant", content=one_shot_output_2),
    Message(role="user", content=usr_input)    
]