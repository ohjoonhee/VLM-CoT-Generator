#!/bin/bash
#SBATCH -J launch               # Job name
#SBATCH -p cpu-farm               # Partition name
#SBATCH --qos=normal      # Quality of Service
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH -o logs/%x_%j.log         # Standard output file path

cd data_viscot/cot_images_tar_split

echo "Working directory: $(pwd)"

cat cot_images_* | tar -xv

echo "Done"