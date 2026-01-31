## Task structure 

This is the previous structure. The tasks are stored as folders on system. We are changing to using database and expand further for production.

task/
├── task_description.txt
├── metadata.json (Only has { metric: "..." })
├── test/
│   ├── labels.csv (Two columns id and label)
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
    │   ├── layout/
    │   │   ├── MainLayout.tsx
    │   │   └── AppBar.tsx
    │   └── feedback/
    │       └── ToastProvider.tsx
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
    │       │   ├── QueryAccordion.tsx
    │       │   ├── ...
    │       │   └── QueryFilesPanel.tsx
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
    ├── types/
    │   └── tasks.types.ts
    │
    ├── utils/
    │   └── noop.ts
    │
    ├── main.tsx
    └── vite-env.d.ts

TODO:
- Add proper display of query name, split type, label in one row. Same goes for task metadata
- Add feedback for task upload/deletion failure
- Add edit tasks (obviously, duh)
- Add a field for task tags (e.g. `cot`, `fewshot`, `graph-level`, `node-level`,...)
- Add filters for task tags