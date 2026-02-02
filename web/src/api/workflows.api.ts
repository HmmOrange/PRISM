import { apiFetch } from "./client";

export interface StartWorkflowPayload {
  task_ids: string[];
  pipeline_path: string;
  rounds: number;
  run_after: boolean;
  calculate_after: boolean;
}

export interface StartWorkflowResponse {
  job_id: string;
}

export type WorkflowJobStatus =
  | "PENDING"
  | "MATERIALIZING"
  | "GENERATING"
  | "DONE"
  | "FAILED";

export interface WorkflowJobStatusResponse {
  status: WorkflowJobStatus;
  current_step: number;
  error: string | null;
}

export function startWorkflowGeneration(
  payload: StartWorkflowPayload
): Promise<StartWorkflowResponse> {
  return apiFetch("/workflows/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getWorkflowGenerationStatus(
  jobId: string
): Promise<WorkflowJobStatusResponse> {
  return apiFetch(`/workflows/generate/${jobId}/status`);
}

export interface WorkflowResult {
  task: string;
  filename: string;
  download_url: string;
}

export function getWorkflowResults(jobId: string): Promise<WorkflowResult[]> {
  return apiFetch(`/workflows/generate/${jobId}/results`);
}
