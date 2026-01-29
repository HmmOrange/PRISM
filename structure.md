## Task structure

task/
├── task_description.txt
├── metadata.json (Only has { metric: "..." })
├── test/
│   ├── labels.csv
│   └── input/
│       ├── 0/
│       │   └── <data files>
│       ├── 1/
│       │   └── <data files>
│       └── ...
└── validation/
    ├── labels.csv
    └── input/
        ├── 0/
        │   └── <data files>
        ├── 1/
        │   └── <data files>
        └── ...

## Project structure

PRISM/
├── api/
│   ├── __init__.py
│   └── task/
│       ├── __init__.py
│       └── task_api.py
│
├── db/
│   ├── base.py
│   ├── session.py
│   ├── settings.py
│   │
│   ├── models/
│   │   └── task/
│   │       ├── __init__.py
│   │       ├── task.py
│   │       ├── dataset_split.py
│   │       └── query.py
│   │
│   ├── schemas/
│   │   └── task/
│   │       ├── __init__.py
│   │       ├── task_schema.py
│   │       ├── dataset_split_schema.py
│   │       └── query_schema.py
│   │
│   └── services/
│       └── task/
│           └── task_service.py
│
├── server/
│   ├── __init__.py
│   ├── tasks_server.py
│   ├── models_server.py
│   ├── api_router.py
│   └── schema_base.py
│
├── storage/
│   ├── __init__.py
│   ├── base_storage.py
│   ├── minio_storage.py
│   └── storage_factory.py
│
├── utils/
│   └── constants.py
│
├── configs/
│   └── config.yaml
│
├── docker/
│   ├── docker-compose.yml
│   ├── entrypoint.sh
│   └── dockerfile/
│       └── tasks_server.Dockerfile
│
├── pyproject.toml
├── poetry.lock
├── LICENSE
└── README.md

## Web

web/
└── src/
    ├── api/
    │   ├── client.ts
    │   └── tasks.api.ts
    │
    ├── app/
    │   ├── App.tsx
    │   └── AppProviders.tsx
    │
    ├── components/
    │   └── layout/
    │       ├── MainLayout.tsx
    │       └── AppBar.tsx
    │
    ├── config/
    │   ├── api.ts
    │   ├── env.ts
    │   └── routes.ts
    │
    ├── features/
    │   └── tasks/
    │       ├── components/
    │       │   ├── TaskForm.tsx
    │       │   └── TaskList.tsx
    │       │
    │       ├── pages/
    │       │   ├── CreateTaskPage.tsx
    │       │   └── TasksPage.tsx
    │       │
    │       ├── types.ts
    │       └── index.ts
    │
    ├── routes/
    │   ├── authed.routes.tsx
    │   ├── public.routes.tsx
    │   └── router.tsx
    │
    ├── styles/
    │   └── theme.ts
    │
    ├── utils/
    │   └── noop.ts
    │
    ├── main.tsx
    └── vite-env.d.ts

Below is a **clean, copy-pasteable project context snapshot**.
This is everything important we’ve established so far, distilled and organized so you (or another LLM) can **resume instantly without re-explaining anything**.

---

# PRISM – Current Project Context (Authoritative)

## 1. High-level goal

Build a **PRISM web system** for creating, managing, and running **benchmark-style AI tasks**, with:

* Clean **microservice backend** (FastAPI, Postgres, MinIO)
* **React web app** for task creation and management
* Strong separation of concerns from day one
* Expandable architecture (tasks → runs → evaluation → PRISM pipelines)

---

## 2. Task format (CRITICAL CONTRACT)

A **task** is a structured bundle of files + metadata.

### Canonical task structure

```
task/
├── task_description.txt
├── metadata.json
├── test/
│   ├── labels.csv
│   └── input/
│       ├── 0/
│       │   └── <data files>
│       ├── 1/
│       │   └── <data files>
│       └── ...
└── validation/
    ├── labels.csv
    └── input/
        ├── 0/
        │   └── <data files>
        ├── 1/
        │   └── <data files>
        └── ...
```

### Semantics

* `task_description.txt`

  * Free-form text describing the task
* `metadata.json`

  * **Exactly one field**

  ```json
  { "metric": "<metric_name>" }
  ```
* `test/` and `validation/` are symmetric
* `labels.csv`

  * Exactly two columns: `id,label`
* Each numeric folder (`0`, `1`, `2`, …) = **one query**
* Query folders may contain **any file types**
* No assumptions about modality (image, audio, text, mixed)

---

## 3. Backend (current state)

### Architecture

* **FastAPI** services
* **Postgres** for metadata
* **MinIO** for file storage
* **Docker Compose** for orchestration
* Config via `configs/config.yaml` (NO env vars in code)

### Backend status

* Tasks service is running in Docker
* Postgres + MinIO running
* Task DB models exist:

  * `Task`
  * `DatasetSplit` (`test`, `validation`)
  * `Query`
* Task creation API exists (metadata only for now)
* File upload flow not wired yet (planned)

---

## 4. Frontend stack (locked in)

### Framework & tooling

* **Vite**
* **React**
* **TypeScript**
* **MUI (Material UI v5)**
* **React Router v6**
* Strict TS settings:

  * `verbatimModuleSyntax: true`
  * Requires `import type` for type-only imports

### Design principles (NON-NEGOTIABLE)

* 1 component / page per file
* No hard-coded routes, API URLs, constants
* Explicit config files for:

  * routes
  * API base
* Separate:

  * authed routes
  * public routes
* Feature-based folder structure
* Files stay small and focused

---

## 5. Frontend project structure (current)

```
web/src/
├── api/
│   ├── client.ts
│   └── tasks.api.ts
│
├── app/
│   ├── App.tsx
│   └── AppProviders.tsx
│
├── components/
│   └── layout/
│       ├── Navbar.tsx
│       └── MainLayout.tsx
│
├── config/
│   ├── api.ts
│   ├── env.ts
│   └── routes.ts
│
├── features/
│   └── tasks/
│       ├── components/
│       │   ├── TaskMetaForm.tsx
│       │   ├── DatasetEditor.tsx
│       │   ├── DatasetColumn.tsx
│       │   └── QueryAccordion.tsx
│       │
│       ├── pages/
│       │   ├── TasksPage.tsx
│       │   └── CreateTaskPage.tsx
│       │
│       ├── types.ts
│       └── index.ts
│
├── routes/
│   ├── public.routes.tsx
│   ├── authed.routes.tsx
│   └── router.tsx
│
├── styles/
│   └── theme.ts
│
├── utils/
│   └── noop.ts
│
├── main.tsx
└── vite-env.d.ts
```

---

## 6. Routing setup

### Route constants

```ts
export const ROUTES = {
  public: {
    home: "/",
    createTask: "/tasks/new",
    login: "/login",
    register: "/register",
  },
  authed: {
    tasks: "/tasks",
    run: "/run",
  },
};
```

### Routing rules

* **Pages are default exports**
* Types imported via:

  ```ts
  import type { RouteObject } from "react-router";
  ```
* Layout applied via `MainLayout` with `<Outlet />`

---

## 7. Navbar (current behavior)

* Left:

  * PRISM (home)
  * Tasks
  * Run (disabled placeholder)
* Right:

  * **Create Task** (primary CTA)
  * Login
  * Register
* Uses route constants (no strings)
* Auth logic NOT implemented yet

---

## 8. MUI theme (current)

* Centralized in `styles/theme.ts`
* Light mode
* Neutral defaults
* No premature over-theming
* Theme injected via `AppProviders`

---

## 9. Task Editor (CURRENT FOCUS)

### Task Editor features (already implemented in UI)

#### Metadata section

* Name
* Metric
* Task description

#### Dataset editor section

* Two columns:

  * **Test**
  * **Validation**
* Each column contains:

  * List of **queries**
* Each query:

  * Has a numeric ID
  * Is **collapsible** (MUI Accordion)
  * Contains a list of uploaded files
  * Supports adding multiple files
* Dataset state is **local React state only** (for now)

### Task editor types

```ts
export type DatasetSplit = "test" | "validation";

export interface QueryFile {
  name: string;
  file: File;
}

export interface QueryData {
  id: number;
  files: QueryFile[];
}

export interface DatasetState {
  test: QueryData[];
  validation: QueryData[];
}
```

### Current limitations (intentional)

* No labels.csv upload yet
* No backend upload yet
* No validation yet
* No stepper / wizard yet
* Editor logs state to console on submit

---

## 10. TypeScript rules (IMPORTANT)

* `verbatimModuleSyntax` is enabled
* **ALL type imports must use `import type`**
* Runtime imports and type imports must be separated
* Pages must have `export default`

---

## 11. What is NOT done yet (planned next)

### Frontend

* Task list UI (GET /tasks)
* Labels.csv upload per split
* Validation (empty queries, missing files)
* Upload → backend → MinIO
* Redirect after successful creation
* Auth pages + protected routes

### Backend

* Multipart upload API
* Labels.csv parsing
* Finalize task endpoint
* Run / execution endpoints

---

## 12. Where we left off (NEXT STEP OPTIONS)

Natural next steps (in order of correctness):

1. **Add labels.csv upload per split** (UI + state)
2. Add **client-side validation** for task editor
3. Wire **Create Task → backend API**
4. Design **upload contract** (presigned URLs / multipart)
5. Add **Task List page UI**

---

This context is **complete and authoritative**.
You can now continue without re-explaining anything.

When you’re ready, just say **what you want to build next**.
