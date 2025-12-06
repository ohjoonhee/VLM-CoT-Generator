#!/bin/bash
#SBATCH -J launch               # Job name
#SBATCH -p cpu-farm               # Partition name
#SBATCH --qos=normal      # Quality of Service
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH -o logs/%x_%j.log         # Standard output file path

PYTHON_FILE=$1

uv run python $PYTHON_FILE