import { apiFetch } from "./client";
import type { TaskListItem, TaskDetail, CreateTaskPayload, CreateTaskResponse } from "../types/tasks.types";

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

/**
 * Get all tasks. 
 */

export function getTasks(): Promise<TaskListItem[]> {
  return apiFetch<TaskListItem[]>("/tasks");
}
/**
 * Get task details by ID.  
 */

export function getTask(taskId: string): Promise<TaskDetail> {
  return apiFetch<TaskDetail>(`/tasks/${taskId}`);
}

export interface CommitFilePayload {
  query_index: number;
  filename: string;
  object_key: string;
  content_type: string;
  size: number;
}

export function commitTaskFiles(
  taskId: string,
  files: CommitFilePayload[]
) {
  return apiFetch(`/tasks/${taskId}/files/commit`, {
    method: "POST",
    body: JSON.stringify({ files }),
  });
}
