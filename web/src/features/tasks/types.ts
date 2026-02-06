/**
 * Task feature local types.
 * Following TypeScript conventions from CODING_STANDARDS.md
 */

export type DatasetSplit = "test" | "validation";

export interface QueryFile {
  id: string;
  file: File;
}

export interface QueryData {
  id: number;
  name: string;
  split: DatasetSplit;
  label: string;
  files: QueryFile[];
}

/**
 * Task creation wizard step labels.
 */
export const WIZARD_STEPS = [
  "Metadata",
  "Pipeline",
  "Dataset",
  "Review",
] as const;

export type WizardStep = (typeof WIZARD_STEPS)[number];

/**
 * Validation error types for task forms.
 */
export interface TaskFormErrors {
  name?: string;
  description?: string;
  metric?: string;
  metrics?: string;
  pipelineTags?: string;
  queries?: string;
}
