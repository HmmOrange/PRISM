# PRISM

PRISM is a platform for ML engineering workflow management, featuring a web interface for task management and a CLI for running ML pipelines.

## Quick Start (Docker)

The fastest way to get started is using Docker Compose, which handles all dependencies and database setup automatically.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)

### 1. Start All Services

```bash
cd docker
docker compose up -d --build
```

This will start:
- **PostgreSQL** database on port `5432`
- **MinIO** object storage on ports `9000` (API) and `9001` (console)
- **Tasks Server** (FastAPI backend) on port `8000` — serves both `/tasks/*` and `/auth/*` endpoints
- **Auth Init** (runs migrations, seeds admin, then exits)

### 2. Wait for Initialization

The services will automatically:
1. Wait for PostgreSQL to be ready
2. Run database migrations
3. **Create the default admin user** (credentials below)

Check the logs to verify everything started correctly:

```bash
docker logs prism_tasks_server
docker logs prism_auth
```

### 3. Start the Frontend (Development)

```bash
cd web
npm install
npm run dev
```

The web app will be available at `http://localhost:5173`

### 4. Login

Default admin credentials:
- **Username:** `admin`
- **Password:** `admin@space`

### Create/Reset Admin User

The admin user is **automatically created during Docker startup** as part of the tasks-server entrypoint. You do not need to run any additional commands.

If the admin already exists, the script will simply log "Admin exists" and continue.

To manually verify or re-run the admin seed script:

```bash
# Via docker exec (recommended)
docker exec prism_tasks_server python scripts-new/create-admin.py

# Or via docker compose
docker compose exec tasks-server python scripts-new/create-admin.py

# For local development (database must be running)
DB_HOST=localhost python scripts-new/create-admin.py
```

The script is idempotent - it will only create the admin if it doesn't exist.

### API Endpoints

The backend API is available at `http://localhost:8000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | Login with username/password |
| `/auth/register` | POST | Create new account |
| `/auth/logout` | POST | Logout (client-side) |
| `/auth/me` | GET | Get current user info |
| `/tasks` | GET | List user's tasks |
| `/tasks` | POST | Create new task |
| `/tasks/{id}` | GET | Get task details |
| `/tasks/{id}` | DELETE | Delete task |

### Stopping Services

```bash
cd docker
docker compose down
```

To also remove all data volumes:

```bash
docker compose down -v
```

---

## Local Development Setup

For development without Docker, or for running ML pipelines:

### Create and activate virtual environment

**Windows:**
```console
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```console
python -m venv venv
source venv/bin/activate
```

### Install PRISM

After activating the virtual environment, you can install `prism` with pip:

```console
pip install -e .
```

For the web server (without ML dependencies):
```console
pip install -e ".[tasks-server]"
```

For ML pipelines:
```console
pip install -e ".[ml]"
```

### Database Configuration

Copy and edit the config file:
```bash
cp configs/config.example.yaml configs/config.yaml
```

Update the database connection settings in `configs/config.yaml`.

### Run Migrations

```bash
alembic upgrade head
```

### Start the Server

```bash
uvicorn server.tasks_server:app --host 0.0.0.0 --port 8000 --reload
```

---


## Dataset

The CA-bench dataset is a collection of 70 CA problems which we use to evaluate the ML engineering capabilities of AI systems.

To install CA problems datasets, run:
```console
prism download -d datasets
```

To install baseline and humand design results, run:

```console
prism download -d results
```

### MLE-bench Lite Evaluation

The Lite dataset contains 21 competitions covering various machine learning tasks. Note that we currently do not support image-to-image tasks, so the `denoising-dirty-documents` competition is excluded from our evaluation.

| Competition ID                              | Category                   | Dataset Size (GB) |
|---------------------------------------------|----------------------------|--------------------|
| aerial-cactus-identification                | Image Classification       | 0.0254            |
| aptos2019-blindness-detection               | Image Classification       | 10.22             |
| detecting-insults-in-social-commentary      | Text Classification        | 0.002             |
| dog-breed-identification                    | Image Classification       | 0.75              |
| dogs-vs-cats-redux-kernels-edition          | Image Classification       | 0.85              |
| histopathologic-cancer-detection            | Image Regression           | 7.76              |
| jigsaw-toxic-comment-classification-challenge | Text Classification        | 0.06              |
| leaf-classification                         | Image Classification       | 0.036             |
| mlsp-2013-birds                             | Audio Classification       | 0.5851            |
| new-york-city-taxi-fare-prediction          | Tabular                   | 5.7               |
| nomad2018-predict-transparent-conductors    | Tabular                   | 0.00624           |
| plant-pathology-2020-fgvc7                  | Image Classification       | 0.8               |
| random-acts-of-pizza                        | Text Classification        | 0.003             |
| ranzcr-clip-catheter-line-classification    | Image Classification       | 13.13             |
| siim-isic-melanoma-classification           | Image Classification       | 116.16            |
| spooky-author-identification                | Text Classification        | 0.0019            |
| tabular-playground-series-dec-2021          | Tabular                   | 0.7               |
| tabular-playground-series-may-2022          | Tabular                   | 0.57              |
| text-normalization-challenge-english-language | Seq->Seq                 | 0.01              |
| text-normalization-challenge-russian-language | Seq->Seq                 | 0.01              |
| the-icml-2013-whale-challenge-right-whale-redux | Audio Classification     | 0.29314           |

## Usage

### Generate workflows from pipeline

To generate workflows from a specific pipeline:

```console
prism generate -p <task_directory> -s <save_directory> -pl <pipeline_path> -n <rounds>
```

Example:
```console
prism generate -p tasks/node-level -s results/my_experiment -pl pipeline/zeroshot_pipeline.py -n 3
```

### Run generated workflows

To run the generated workflows:

```console
prism run -p <task_directory> -s <save_directory> -n <rounds>
```

Example:
```console
prism run -p tasks/node-level -s results/my_experiment -n 3
```

### Calculate solution scores

To calculate scores for executed solutions:

```console
prism calculate -p <task_directory> -s <save_directory> -n <rounds>
```

Example:
```console
prism calculate -p tasks/node-level -s results/my_experiment -n 3
```

### Run complete pipeline

To generate, run and calculate scores in a single command:

```console
prism generate -p <task_directory> -s <save_directory> -pl <pipeline_path> -n <rounds> --run-after --calculate-after
```

Example:
```console
prism generate -p tasks/node-level -s results/my_experiment -pl pipeline/zeroshot_pipeline.py -n 3 --run-after --calculate-after
```

### Main parameters

- `-p, --path`: Path to task directory (multiple tasks supported)
- `-s, --save-dir`: Directory to save results (must be a subfolder of 'results/')
- `-pl, --pipeline_path`: Path to pipeline for generating solutions
- `-n, --rounds`: Number of rounds to run (default: 1)
- `--run-after`: Run workflows immediately after generation
- `--calculate-after`: Calculate scores after running (requires --run-after)

### List available datasets

```console
prism download --list
```
