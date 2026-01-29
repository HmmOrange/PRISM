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

export interface PresignedFile {
  filename: string;
  upload_url: string;
  object_key: string;
}

export interface QueryUpload {
  query_index: number;
  files: PresignedFile[];
}


export interface CreateTaskResponse {
  task_id: string;
  uploads: QueryUpload[];
}

export interface QueryFile {
  filename: string;
  object_key: string;
  content_type: string;
  size: number;
}

export interface QueryDetail {
  index: number;
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