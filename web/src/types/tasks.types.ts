/**
 * Task-related type definitions.
 * Following TypeScript strict mode conventions from CODING_STANDARDS.md
 */

export interface TaskListItem {
  id: string;
  name: string;
  description: string;
  metric: string;
  metrics?: string[]; // Support for multi-select metrics

  total_queries: number;
  test_queries: number;
  validation_queries: number;

  pipeline_tags?: string[]; // Pipeline configuration tags

  created_at: string;
  updated_at?: string;
}

export interface CreateTaskPayload {
  name: string;
  metric: string;
  metrics?: string[]; // Multi-select metrics
  description: string;
  pipeline_tags?: string[]; // Pipeline configuration tags
  queries: {
    id: number;
    name: string;
    split: "test" | "validation";
    label: string;
    files: {
      filename: string;
      content_type: string;
    }[];
  }[];
}

export interface PresignedPost {
  filename: string;
  object_key: string;
  url: string;
  fields: Record<string, string>;
}

export interface QueryUpload {
  query_index: number;
  files: PresignedPost[];
}

export interface CreateTaskResponse {
  task_id: string;
  uploads: QueryUpload[];
}

export interface QueryFile {
  id: string;
  filename: string;
  object_key: string;
  content_type: string;
  size: number;
  download_url: string;
}

export interface QueryDetail {
  index: number; // 0..N-1
  name: string;
  split: "test" | "validation";
  label: string;
  files: QueryFile[];
}

export interface TaskDetail {
  id: string;
  name: string;
  description: string;
  metric: string;
  metrics?: string[]; // Multi-select metrics
  pipeline_tags?: string[]; // Pipeline configuration tags
  queries: QueryDetail[];
  created_at?: string;
  updated_at?: string;
}

export interface LocalQueryFile {
  id: string;
  file: File;
}

export interface ExistingQueryFile {
  id: string;
  filename: string;
  size: number;
  download_url: string;
  content_type: string;
  object_key: string;  // Needed for preservation on update
}

export interface EditableQuery {
  id: number; // 0..N-1
  name: string;
  split: "test" | "validation";
  label: string;
  files: LocalQueryFile[];
  existingFiles?: ExistingQueryFile[]; // Files already on the server
}

export interface UpdateTaskPayload {
  name: string;
  metric: string;
  metrics?: string[];
  description: string;
  pipeline_tags?: string[];
  queries: {
    id: number;
    name: string;
    split: "test" | "validation";
    label: string;
    files: {
      filename: string;
      content_type: string;
    }[];
    existing_files: string[];  // object_keys to preserve
  }[];
}
