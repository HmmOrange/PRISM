# PRISM Porting Guide

**Purpose:** Comprehensive reference for reconstructing the backend and porting the `web/src` frontend to another project.

**Last Updated:** February 2026

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Database Schema](#2-database-schema)
3. [Backend API Endpoints](#3-backend-api-endpoints)
4. [Authentication System](#4-authentication-system)
5. [File Storage System](#5-file-storage-system)
6. [Frontend Architecture](#6-frontend-architecture)
7. [Frontend-Backend Contracts](#7-frontend-backend-contracts)
8. [Configuration & Environment](#8-configuration--environment)
9. [Migration Notes](#9-migration-notes)

---

## 1. System Architecture Overview

### Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend Framework** | FastAPI (Python 3.10+) |
| **Database** | PostgreSQL 15 |
| **ORM** | SQLAlchemy 2.x |
| **Migrations** | Alembic |
| **Object Storage** | MinIO (S3-compatible) |
| **Authentication** | JWT (python-jose) + bcrypt |
| **Frontend** | React 19 + TypeScript + Vite |
| **UI Library** | MUI (Material UI) v7 |
| **Routing** | React Router DOM v7 |

### Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Vite)                       │
│                     http://localhost:5173                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend (FastAPI Server)                    │
│                     http://localhost:8000                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Auth API │  │ Task API │  │Storage API│  │ Workflow API │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└─────────────────────────────┬───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   PostgreSQL     │ │      MinIO       │ │   ML Pipeline    │
│   (port 5432)    │ │   (port 9000)    │ │   (optional)     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### Docker Services

```yaml
# docker/docker-compose.yml
services:
  tasks-server:    # FastAPI backend on port 8000
  auth:            # Auth service (shared DB)
  postgres:        # PostgreSQL on port 5432
  minio:           # MinIO on ports 9000 (API), 9001 (Console)
```

---

## 2. Database Schema

### Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                          users                               │
├─────────────────────────────────────────────────────────────┤
│ id            UUID PRIMARY KEY                              │
│ username      VARCHAR(150) UNIQUE NOT NULL                  │
│ email         VARCHAR(255) UNIQUE NOT NULL                  │
│ hashed_password VARCHAR(255) NOT NULL                       │
│ is_active     BOOLEAN DEFAULT TRUE                          │
│ created_at    TIMESTAMP WITH TIMEZONE                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                          tasks                               │
├─────────────────────────────────────────────────────────────┤
│ id            UUID PRIMARY KEY                              │
│ name          VARCHAR(255) NOT NULL                         │
│ description   TEXT NOT NULL                                 │
│ metric        VARCHAR(128) NOT NULL                         │
│ pipeline_tags TEXT[] (PostgreSQL ARRAY)                     │
│ user_id       UUID FK → users.id (CASCADE DELETE)           │
│ created_at    TIMESTAMP WITH TIMEZONE                       │
│ updated_at    TIMESTAMP WITH TIMEZONE                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         queries                              │
├─────────────────────────────────────────────────────────────┤
│ id            INTEGER PRIMARY KEY AUTO                      │
│ index         INTEGER NOT NULL (0..N-1)                     │
│ task_id       UUID FK → tasks.id (CASCADE DELETE)           │
│ name          VARCHAR(256) DEFAULT ""                       │
│ split         VARCHAR(32) NOT NULL ("test"|"validation")    │
│ label         VARCHAR DEFAULT ""                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       query_files                            │
├─────────────────────────────────────────────────────────────┤
│ id            UUID PRIMARY KEY                              │
│ query_id      INTEGER FK → queries.id (CASCADE DELETE)      │
│ filename      VARCHAR NOT NULL                              │
│ object_key    VARCHAR NOT NULL UNIQUE                       │
│ content_type  VARCHAR NOT NULL                              │
│ size          BIGINT NOT NULL                               │
└─────────────────────────────────────────────────────────────┘
```

### SQLAlchemy Models

**Location:** `db/models/`

| Model | File | Table |
|-------|------|-------|
| `UserModel` | `db/models/user/user.py` | `users` |
| `TaskModel` | `db/models/task/task.py` | `tasks` |
| `QueryModel` | `db/models/task/query.py` | `queries` |
| `QueryFileModel` | `db/models/task/query_file.py` | `query_files` |

### Alembic Migrations

**Location:** `alembic/versions/`

| Revision | Description |
|----------|-------------|
| `f4691d8c6fd1` | Baseline - tasks, queries, query_files |
| `003_...` | Add users table and task.user_id FK |
| `004_...` | Add pipeline_tags array to tasks |
| `005_...` | Add updated_at to tasks |
| `001_...` | Add name column to queries |

---

## 3. Backend API Endpoints

### Router Structure

**Main Router:** `server/api_router.py`

```python
router.include_router(auth_router)           # /auth/*
router.include_router(task_router)           # /tasks/*
router.include_router(storage_router)        # /storage/*
router.include_router(workflow_results_router)  # /workflows/*
router.include_router(workflow_router)       # /workflows/* (optional - ML deps)
```

---

### 3.1 Authentication API (`/auth`)

**File:** `api/auth/__init__.py`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/login` | Authenticate user, return JWT | No |
| POST | `/auth/register` | Create new account, return JWT | No |
| POST | `/auth/logout` | Logout (client-side token removal) | No |
| GET | `/auth/me` | Get current authenticated user | Yes |

#### Login Request/Response

```typescript
// Request
interface LoginRequest {
  username: string;  // min 1, max 150 chars
  password: string;  // min 4 chars
}

// Response
interface AuthResponse {
  user: {
    id: string;
    username: string;
    email: string;
    created_at: string;  // ISO datetime
  };
  access_token: string;
  token_type: "bearer";
}
```

#### Register Request

```typescript
interface RegisterRequest {
  username: string;  // min 3, max 150 chars
  email: string;     // max 255 chars
  password: string;  // min 8 chars
}
```

---

### 3.2 Tasks API (`/tasks`)

**File:** `api/task/task_api.py`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/tasks` | Create task, get presigned upload URLs | Yes |
| GET | `/tasks` | List all tasks for current user | Yes |
| GET | `/tasks/{task_id}` | Get task details with queries/files | Yes |
| PUT | `/tasks/{task_id}` | Full update task (replaces queries) | Yes |
| PATCH | `/tasks/{task_id}` | Partial update metadata only | Yes |
| DELETE | `/tasks/{task_id}` | Delete task and all associated data | Yes |
| POST | `/tasks/{task_id}/files/commit` | Commit uploaded file metadata | Yes |
| POST | `/tasks/import/zip` | Create task from ZIP file | Yes |

#### Create Task Request

```typescript
interface CreateTaskPayload {
  name: string;
  description: string;
  metric: string;
  pipeline_tags?: string[];
  queries: {
    id: number;              // 0-indexed
    name: string;
    split: "test" | "validation";
    label: string;           // required for validation, optional for test
    files: {
      filename: string;
      content_type: string;  // MIME type
    }[];
  }[];
}
```

#### Create Task Response

```typescript
interface CreateTaskResponse {
  task_id: string;           // UUID
  uploads: {
    query_index: number;
    files: {
      filename: string;
      object_key: string;    // MinIO storage path
      url: string;           // Presigned POST URL
      fields: Record<string, string>;  // Form fields for POST
    }[];
  }[];
}
```

#### Task List Response

```typescript
interface TaskListItem {
  id: string;
  name: string;
  description: string;
  metric: string;
  pipeline_tags?: string[];
  total_queries: number;
  test_queries: number;
  validation_queries: number;
  created_at: string;
}
```

#### Task Detail Response

```typescript
interface TaskDetail {
  id: string;
  name: string;
  description: string;
  metric: string;
  pipeline_tags?: string[];
  queries: {
    index: number;
    name: string;
    split: "test" | "validation";
    label: string;
    files: {
      filename: string;
      object_key: string;
      content_type: string;
      size: number;
      download_url: string;  // Backend proxy URL
    }[];
  }[];
  created_at: string;
  updated_at: string;
}
```

#### Update Task Request (PUT)

```typescript
interface UpdateTaskPayload {
  name: string;
  description: string;
  metric: string;
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
```

#### Partial Update Request (PATCH)

```typescript
interface TaskMetadataUpdate {
  name?: string;
  description?: string;
  metric?: string;
  pipeline_tags?: string[];
}
```

#### Commit Files Request

```typescript
interface CommitFilesRequest {
  files: {
    query_index: number;
    filename: string;
    object_key: string;
    content_type: string;
    size: number;
  }[];
}
```

---

### 3.3 Storage API (`/storage`)

**File:** `api/storage/storage_api.py`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/storage/download?object_key=...` | Download file from MinIO | No |

Returns file stream with appropriate `Content-Type` header.

---

### 3.4 Workflows API (`/workflows`)

**Files:** `api/workflow/workflow_api.py`, `api/workflow/workflow_results_api.py`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/workflows/generate` | Start workflow generation job |
| GET | `/workflows/generate/{job_id}/status` | Get job status |
| GET | `/workflows/generate/{job_id}/results` | List generated files |
| GET | `/workflows/generate/{job_id}/results/{file_path}` | Download file |

#### Generate Workflow Request

```typescript
interface StartWorkflowPayload {
  task_ids: string[];
  pipeline_path: string;
  rounds: number;
  run_after: boolean;
  calculate_after: boolean;
}
```

#### Job Status Response

```typescript
interface WorkflowJobStatusResponse {
  status: "PENDING" | "MATERIALIZING" | "GENERATING" | "DONE" | "FAILED";
  current_step: number;
  error: string | null;
}
```

---

## 4. Authentication System

### JWT Configuration

**File:** `db/services/user/auth_service.py`

```python
JWT_SECRET_KEY = "prism-secret-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
```

### Password Hashing

- **Algorithm:** bcrypt
- **Rounds:** 12
- **Truncation:** 72 bytes (bcrypt limit)

### Auth Flow

1. **Login/Register:** Client sends credentials → Server validates → Returns JWT
2. **Authenticated Requests:** Client includes `Authorization: Bearer <token>` header
3. **Token Validation:** `api/deps.py` → `get_current_user()` dependency extracts and validates JWT
4. **Token Storage (Frontend):**
   - `rememberMe: true` → `localStorage.prism_auth_token`
   - `rememberMe: false` → `sessionStorage.prism_auth_token`

### Auth Guard Implementation (Frontend)

**File:** `web/src/features/auth/components/AuthGuard.tsx`

```tsx
// Redirects to /login if not authenticated
// Shows loading spinner while checking auth state
```

**File:** `web/src/features/auth/components/GuestGuard.tsx`

```tsx
// Redirects to /dashboard if already authenticated
// Used on login/register pages
```

---

## 5. File Storage System

### MinIO Configuration

**File:** `utils/constants.py`

```python
MINIO_EXTERNAL_ENDPOINT = "localhost:9000"    # Client uploads
MINIO_INTERNAL_ENDPOINT = "minio:9000"        # Server operations (Docker network)
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "prism"
MINIO_SECURE = False
```

### Object Key Structure

```
{task_id}/{split}/input/{query_index}/{filename}

Example:
550e8400-e29b-41d4-a716-446655440000/validation/input/0/image.jpg
```

### Upload Flow

1. **Create Task:** Backend generates presigned POST URLs for each file
2. **Client Upload:** Frontend uploads directly to MinIO using presigned URL + fields
3. **Commit Files:** Frontend calls `/tasks/{id}/files/commit` with file metadata
4. **Download:** Frontend requests `/storage/download?object_key=...` (backend proxies)

### MinIO Storage Class

**File:** `storage/minio_storage.py`

```python
class MinIOStorage:
    def generate_presigned_upload_post(object_key, content_type, expires_in=3600)
    def get_object_stream(object_key)
    def upload_file(object_key, file_path, content_type)
```

---

## 6. Frontend Architecture

### Directory Structure

```
web/src/
├── api/                    # API client and endpoint functions
│   ├── client.ts           # Base fetch wrapper with auth
│   ├── tasks.api.ts        # Task CRUD operations
│   └── workflows.api.ts    # Workflow generation
├── app/                    # App-level components
├── assets/                 # Static assets
├── components/             # Shared UI components
│   ├── common/             # Tag, FormField
│   ├── feedback/           # Toasts, alerts
│   ├── inputs/             # Form inputs
│   └── layout/             # MainLayout, NavBar
├── config/                 # Configuration constants
│   ├── api.ts              # API base URL
│   ├── env.ts              # Environment variables
│   ├── metrics.ts          # Available evaluation metrics
│   ├── routes.ts           # Route path definitions
│   └── taskTypes.ts        # Pipeline tag categories
├── features/               # Feature modules
│   ├── auth/               # Authentication
│   ├── dashboard/          # Landing dashboard
│   ├── tasks/              # Task management
│   └── workflows/          # Workflow generation
├── routes/                 # React Router configuration
│   ├── authed.routes.tsx   # Protected routes
│   ├── public.routes.tsx   # Guest routes
│   └── router.tsx          # Main router
├── styles/                 # Global styles
├── types/                  # Shared TypeScript types
│   └── tasks.types.ts      # Task-related interfaces
├── utils/                  # Utility functions
└── main.tsx                # App entry point
```

### Dependencies

```json
{
  "dependencies": {
    "@emotion/react": "^11.14.0",
    "@emotion/styled": "^11.14.1",
    "@fontsource/outfit": "^5.2.8",
    "@mui/icons-material": "^7.3.7",
    "@mui/material": "^7.3.7",
    "lucide-react": "^0.563.0",
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "react-router-dom": "^7.13.0"
  }
}
```

### Route Configuration

```typescript
// Public routes (guest only)
/login          → LoginPage
/register       → RegisterPage

// Protected routes (require auth)
/dashboard      → DashboardPage
/tasks          → TaskLibraryPage
/tasks/new      → CreateTaskWizardPage
/tasks/:taskId  → TaskDetailPage
/run            → RunPage
```

### Feature Module Structure

Each feature follows this pattern:

```
features/{feature}/
├── api.ts              # Feature-specific API calls
├── types.ts            # TypeScript interfaces
├── index.ts            # Public exports
├── context/            # React context (if needed)
├── components/         # UI components
├── hooks/              # Custom hooks
└── pages/              # Page components
```

### Auth Feature

**Exports:** `AuthProvider`, `useAuth`, `AuthGuard`, `GuestGuard`

```typescript
// Context value
interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (credentials: RegisterCredentials) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}
```

### Tasks Feature Components

| Component | Purpose |
|-----------|---------|
| `CreateTaskWizardPage` | Multi-step task creation wizard |
| `TaskLibraryPage` | List/filter/search tasks |
| `TaskDetailPage` | View/edit task details |
| `MetadataStep` | Wizard step 1: name, description, metrics |
| `PipelineStep` | Wizard step 2: pipeline tag selection |
| `DatasetStep` | Wizard step 3: query/file management |
| `ReviewStep` | Wizard step 4: summary before creation |
| `QueryAccordion` | Collapsible query editor |
| `SectionCard` | Card layout for detail sections |
| `TaskCard` | Task preview card for library |

---

## 7. Frontend-Backend Contracts

### API Client

**File:** `web/src/api/client.ts`

```typescript
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  // Auto-sets Content-Type: application/json (unless FormData)
  // Auto-attaches Bearer token from storage
  // Throws on non-2xx responses
  // Returns parsed JSON
}
```

### Environment Variable

```typescript
// web/src/config/env.ts
VITE_API_BASE_URL  // e.g., "http://localhost:8000"
```

### Token Storage Keys

```typescript
const AUTH_TOKEN_KEY = "prism_auth_token";
// Stored in localStorage (remember me) or sessionStorage
```

### Error Response Format

```typescript
// Backend returns:
{ "detail": "Error message" }

// Frontend throws:
new Error(`API error ${status}: ${detail}`)
```

---

## 8. Configuration & Environment

### Backend Configuration

**File:** `configs/config.yaml`

```yaml
db:
  host: postgres
  port: 5432
  name: prism
  user: prism
  password: prism

storage:
  minio:
    external_endpoint: localhost:9000
    internal_endpoint: minio:9000
    access_key: minioadmin
    secret_key: minioadmin
    bucket: prism
    secure: false

server:
  host: 0.0.0.0
  port: 8000

cors:
  allow_origins:
    - http://localhost:5173
    - http://localhost:3000
  allow_methods: ["*"]
  allow_headers: ["*"]
  allow_credentials: true
```

### Environment Variables (Override config.yaml)

```bash
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

### Frontend Environment

**File:** `web/.env`

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 9. Migration Notes

### Backend Reconstruction Checklist

1. **Database Setup**
   - [ ] PostgreSQL 15+ with UUID extension
   - [ ] Run Alembic migrations: `alembic upgrade head`
   - [ ] Seed admin user (username: `admin`, password: `admin`)

2. **Storage Setup**
   - [ ] MinIO or S3-compatible storage
   - [ ] Create bucket (default: `prism`)
   - [ ] Configure CORS for browser uploads

3. **Server Setup**
   - [ ] FastAPI with CORS middleware
   - [ ] Mount routers: auth, tasks, storage, workflows
   - [ ] JWT dependency for protected routes

4. **Critical Services**
   - [ ] `db/services/user/auth_service.py` - Password hashing, JWT
   - [ ] `db/services/task/task_service.py` - Task CRUD with presigned URLs
   - [ ] `db/services/task/file_service.py` - File commit logic
   - [ ] `storage/minio_storage.py` - Presigned URL generation

### Frontend Porting Checklist

1. **Dependencies**
   - [ ] React 19, React Router DOM v7
   - [ ] MUI v7 with Emotion
   - [ ] Vite with TypeScript

2. **Configuration**
   - [ ] Copy `web/src/config/` folder
   - [ ] Set `VITE_API_BASE_URL` environment variable

3. **Core Files to Port**
   - [ ] `api/client.ts` - API client with auth
   - [ ] `features/auth/` - Full auth module
   - [ ] `routes/` - Router configuration
   - [ ] `components/layout/` - MainLayout, NavBar

4. **Feature Modules**
   - [ ] `features/tasks/` - Task management (largest module)
   - [ ] `features/dashboard/` - Landing page
   - [ ] `features/workflows/` - Workflow generation

### Breaking Changes to Note

1. **Auth Token Key:** `prism_auth_token` - Change if needed
2. **Object Key Format:** `{task_id}/{split}/input/{query_index}/{filename}`
3. **Pipeline Tags:** Stored as PostgreSQL `TEXT[]` array
4. **Metrics:** Single string (backend), but frontend supports multi-select UI
5. **Query Index:** 0-indexed, used as unique identifier within task

### Available Evaluation Metrics

```typescript
const AVAILABLE_METRICS = [
  "accuracy", "f1", "rouge", "r2", "code_bleu",
  "numerical_accuracy", "semantic_similarity", "semantic_word_similarity"
];
```

### Pipeline Tag Categories

```typescript
const TASK_CATEGORIES = [
  "Multimodal", "Computer Vision", "Natural Language Processing",
  "Audio", "Tabular", "Reinforcement Learning"
];
```

---

## Appendix: File References

| Purpose | Backend File | Frontend File |
|---------|--------------|---------------|
| Auth API | `api/auth/__init__.py` | `features/auth/api.ts` |
| Auth Types | `db/schemas/user/auth_schema.py` | `features/auth/types.ts` |
| Task API | `api/task/task_api.py` | `api/tasks.api.ts` |
| Task Types | `db/schemas/task/task_schema.py` | `types/tasks.types.ts` |
| Storage | `storage/minio_storage.py` | (direct MinIO upload) |
| Workflows | `api/workflow/workflow_api.py` | `api/workflows.api.ts` |
| DB Session | `db/session.py` | N/A |
| Auth Dependency | `api/deps.py` | `api/client.ts` |
| Configuration | `configs/config.yaml` | `config/*.ts` |
