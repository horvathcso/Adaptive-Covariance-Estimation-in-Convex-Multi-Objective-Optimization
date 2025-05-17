#!/bin/bash
#SBATCH --job-name=aco_active_learning
#SBATCH --output=aco_active_learning_%j.out
#SBATCH --error=aco_active_learning_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Load necessary modules (uncomment and modify as needed)
# module load python/3.10

# Activate virtual environment if needed
source .venv/bin/activate

# Run the Python script
python aco_active_learning.py