/**
 * Available metrics for task evaluation.
 * These should match the metrics used in the backend /tasks folders.
 */
export const AVAILABLE_METRICS = [
  { value: "accuracy", label: "Accuracy" },
  { value: "f1", label: "F1 Score" },
  { value: "rouge", label: "ROUGE" },
  { value: "r2", label: "R² (Coefficient of Determination)" },
  { value: "code_bleu", label: "CodeBLEU" },
  { value: "numerical_accuracy", label: "Numerical Accuracy" },
  { value: "semantic_similarity", label: "Semantic Similarity" },
  { value: "semantic_word_similarity", label: "Semantic Word Similarity" },
] as const;

export type MetricValue = typeof AVAILABLE_METRICS[number]["value"];

/**
 * Get the display label for a metric value.
 * Falls back to the raw value if not found.
 */
export function getMetricLabel(value: string): string {
  const metric = AVAILABLE_METRICS.find((m) => m.value === value);
  return metric ? metric.label : value;
}
