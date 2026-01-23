# VLM-CoT-Generator

A research repository for generating and refining Vision-Language Model (VLM) Chain-of-Thought (CoT) datasets.

## Project Structure

This repository follows a research-oriented structure, separating library code, experiment scripts, and configurations.

```text
.
├── configs/            # Global configuration files and prompts
├── data/               # Raw and processed datasets (ignored by git)
├── output/             # Model predictions and evaluation results (ignored by git)
├── scripts/            # Main entrypoints and experiment scripts
│   ├── data/           # Data management and upload scripts
│   ├── eval/           # Model evaluation and judging scripts
│   ├── inference/      # Model inference and generation scripts
│   └── tools/          # Utility tools and viewers
├── src/                # Core library modules
│   ├── data_prep/      # Dataset-specific processing logic (COCO, Object365)
│   ├── eval/           # Shared evaluation metrics and logic
│   ├── inference/      # Model API wrappers and base inference logic
│   └── utils/          # General helper functions and utilities
├── tmp/                # Temporary files and experimental code
├── pyproject.toml      # Project dependencies and metadata
└── README.md           # This file
```

## Setup

This project uses `uv` for dependency management.

```bash
uv sync
```

## Key Workflows

### Inference
Generate CoT reasoning using various models:
```bash
python scripts/inference/run_fireworks.py
```

### Evaluation
Judge model predictions:
```bash
python scripts/eval/run_openai_judge.py
```

### Data Management
Push processed datasets to Hugging Face Hub:
```bash
python scripts/data/run_uploader.py
```
