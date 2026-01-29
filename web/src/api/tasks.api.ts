import { apiFetch } from "./client";

/* =========================
   Request types
   ========================= */

export interface CreateTaskPayload {
  name: string;
  metric: string;
  description: string;
  queries: {
    id: number;
    split: "test" | "validation";
    label: string;
    files: {
      filename: string;
      content_type: string;
    }[];
  }[];
}

/* =========================
   Response types
   ========================= */

export interface PresignedFile {
  filename: string;
  upload_url: string;
  object_key: string;
}

export interface QueryUpload {
  query_id: number;
  files: PresignedFile[];
}

export interface CreateTaskResponse {
  task_id: number;
  uploads: QueryUpload[];
}

/* =========================
   API calls
   ========================= */

/**
 * Create a task and receive presigned upload URLs.
 * Does NOT upload files.
 */
export async function createTask(
  payload: CreateTaskPayload
): Promise<CreateTaskResponse> {
  return apiFetch<CreateTaskResponse>("/tasks", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
