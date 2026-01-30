export interface TaskListItem {
  id: string;
  name: string;
  description: string;
  metric: string;

  total_queries: number;
  test_queries: number;
  validation_queries: number;

  created_at: string;
}

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
  split: "test" | "validation";
  label: string;
  files: QueryFile[];
}

export interface TaskDetail {
  id: string;
  name: string;
  description: string;
  metric: string;
  queries: QueryDetail[];
}

export interface LocalQueryFile {
  id: string;
  file: File;
}

export interface EditableQuery {
  id: number; // 0..N-1
  name: string;
  split: "test" | "validation";
  label: string;
  files: LocalQueryFile[];
}
