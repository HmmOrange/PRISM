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
