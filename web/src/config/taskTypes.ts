/**
 * Task Types and Pipeline Tags configuration.
 * Based on TASK_TYPES_AND_PIPELINES.md
 * Implements SRS 2.3.1 Stage 2: Pipeline Configuration
 */

export interface TaskCategory {
  id: string;
  name: string;
  tasks: string[];
}

export const TASK_CATEGORIES: TaskCategory[] = [
  {
    id: "multimodal",
    name: "Multimodal",
    tasks: [
      "Audio-Text-to-Text",
      "Image-Text-to-Text",
      "Image-Text-to-Image",
      "Image-Text-to-Video",
      "Visual Question Answering",
      "Document Question Answering",
      "Video-Text-to-Text",
      "Visual Document Retrieval",
      "Any-to-Any",
    ],
  },
  {
    id: "computer-vision",
    name: "Computer Vision",
    tasks: [
      "Depth Estimation",
      "Image Classification",
      "Object Detection",
      "Image Segmentation",
      "Text-to-Image",
      "Image-to-Text",
      "Image-to-Image",
      "Image-to-Video",
      "Unconditional Image Generation",
      "Video Classification",
      "Text-to-Video",
      "Zero-Shot Image Classification",
      "Mask Generation",
      "Zero-Shot Object Detection",
      "Text-to-3D",
      "Image-to-3D",
      "Image Feature Extraction",
      "Keypoint Detection",
      "Video-to-Video",
    ],
  },
  {
    id: "nlp",
    name: "Natural Language Processing",
    tasks: [
      "Text Classification",
      "Token Classification",
      "Table Question Answering",
      "Question Answering",
      "Zero-Shot Classification",
      "Translation",
      "Summarization",
      "Feature Extraction",
      "Text Generation",
      "Fill-Mask",
      "Sentence Similarity",
      "Text Ranking",
    ],
  },
  {
    id: "audio",
    name: "Audio",
    tasks: [
      "Text-to-Speech",
      "Text-to-Audio",
      "Automatic Speech Recognition",
      "Audio-to-Audio",
      "Audio Classification",
      "Voice Activity Detection",
    ],
  },
  {
    id: "tabular",
    name: "Tabular",
    tasks: [
      "Tabular Classification",
      "Tabular Regression",
      "Time Series Forecasting",
    ],
  },
  {
    id: "reinforcement-learning",
    name: "Reinforcement Learning",
    tasks: [
      "Reinforcement Learning",
      "Robotics",
    ],
  },
  {
    id: "other",
    name: "Other",
    tasks: [
      "Graph Machine Learning",
    ],
  },
];

/**
 * Flatten all task types into a single array.
 */
export function getAllTaskTypes(): string[] {
  return TASK_CATEGORIES.flatMap((cat) => cat.tasks);
}

/**
 * Get category by task name.
 */
export function getCategoryByTask(taskName: string): TaskCategory | undefined {
  return TASK_CATEGORIES.find((cat) => cat.tasks.includes(taskName));
}
