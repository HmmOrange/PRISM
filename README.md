# PRISM

```
cd docker
docker compose up -d --build
docker exec -it prism_tasks_server alembic upgrade head
```

## Setup

### Create and activate virtual environment

Before installation, create and activate a virtual environment:

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
