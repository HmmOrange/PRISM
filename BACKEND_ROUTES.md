# PRISM Backend API Routes

This document describes all available API endpoints in the PRISM backend.

## Base URL

Development: `http://localhost:8000`

---

## Tasks API

All task-related endpoints for CRUD operations on ML tasks.

### Create Task

**POST** `/tasks`

Creates a new task and returns presigned upload URLs for files.

**Request Body:**
```json
{
  "name": "string (required)",
  "description": "string (required)",
  "metric": "string (required)",
  "queries": [
    {
      "id": "int (0-indexed)",
      "name": "string (optional)",
      "split": "\"test\" | \"validation\"",
      "label": "string (optional for test, required for validation)",
      "files": [
        {
          "filename": "string",
          "content_type": "string (MIME type)"
        }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "task_id": "uuid",
  "uploads": [
    {
      "query_index": "int",
      "files": [
        {
          "filename": "string",
          "object_key": "string (storage path)",
          "url": "string (presigned upload URL)",
          "fields": { "key": "value" }
        }
      ]
    }
  ]
}
```

---

### List Tasks

**GET** `/tasks`

Returns all tasks with summary information.

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "string",
    "description": "string",
    "metric": "string",
    "total_queries": "int",
    "test_queries": "int",
    "validation_queries": "int",
    "created_at": "datetime"
  }
]
```

---

### Get Task Detail

**GET** `/tasks/{task_id}`

Returns detailed information about a specific task including queries and files.

**Parameters:**
- `task_id` (path): UUID of the task

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "metric": "string",
  "queries": [
    {
      "index": "int",
      "name": "string",
      "split": "string",
      "label": "string",
      "files": [
        {
          "filename": "string",
          "object_key": "string",
          "content_type": "string",
          "size": "int",
          "download_url": "string"
        }
      ]
    }
  ]
}
```

---

### Update Task

**PUT** `/tasks/{task_id}`

Updates an existing task.

**Parameters:**
- `task_id` (path): UUID of the task

**Request Body:**
```json
{
  "name": "string",
  "description": "string",
  "metric": "string",
  "queries": [
    {
      "id": "int",
      "name": "string",
      "split": "string",
      "label": "string",
      "files": [
        {
          "filename": "string",
          "content_type": "string"
        }
      ]
    }
  ]
}
```

---

### Delete Task

**DELETE** `/tasks/{task_id}`

Deletes a task and all associated queries/files.

**Parameters:**
- `task_id` (path): UUID of the task

**Response:** `204 No Content`

---

### Commit Files

**POST** `/tasks/{task_id}/files/commit`

Commits uploaded file metadata after files have been uploaded to storage.

**Parameters:**
- `task_id` (path): UUID of the task

**Request Body:**
```json
{
  "files": [
    {
      "query_index": "int",
      "filename": "string",
      "object_key": "string",
      "content_type": "string",
      "size": "int"
    }
  ]
}
```

---

### Import Task from ZIP

**POST** `/tasks/import/zip`

Creates a task from a ZIP file upload.

**Request:** `multipart/form-data`
- `zip_file`: The ZIP file containing task data

**Response:** Same as Get Task Detail

---

## Workflows API

Endpoints for generating and managing ML workflows.

### Generate Workflows

**POST** `/workflows/generate`

Starts an asynchronous workflow generation job.

**Request Body:**
```json
{
  "task_ids": ["uuid", ...],
  "pipeline_path": "string (e.g., 'prism_pipeline.py')",
  "rounds": "int (default: 1)",
  "run_after": "boolean (default: false)",
  "calculate_after": "boolean (default: false)"
}
```

**Response:**
```json
{
  "job_id": "string"
}
```

---

### Get Generation Status

**GET** `/workflows/generate/{job_id}/status`

Gets the status of a workflow generation job.

**Parameters:**
- `job_id` (path): Job identifier

**Response:**
```json
{
  "status": "pending | running | completed | failed",
  "progress": "int (0-100)",
  "message": "string (optional)",
  "error": "string (optional)"
}
```

---

### Get Workflow Results

**GET** `/workflows/generate/{job_id}/results`

Lists generated workflow files for a completed job.

**Parameters:**
- `job_id` (path): Job identifier

**Response:**
```json
[
  {
    "task": "string (task folder name)",
    "relative_path": "string",
    "filename": "string",
    "download_url": "string"
  }
]
```

---

### Download Workflow File

**GET** `/workflows/generate/{job_id}/results/{file_path}`

Downloads a specific generated workflow file.

**Parameters:**
- `job_id` (path): Job identifier
- `file_path` (path): Relative path to the file

**Response:** Python file content (`text/x-python`)

---

## Storage API

Endpoints for file storage operations.

### Download File

**GET** `/storage/download`

Downloads a file from object storage.

**Query Parameters:**
- `object_key` (required): The storage path of the file

**Response:** File stream with appropriate content-type

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Error description"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error description"
}
```

---

## Authentication

Currently, the API does not require authentication. This will be updated in future versions per SRS 2.1.

---

## CORS

The API allows cross-origin requests from the frontend development server (default: `http://localhost:5173`).
