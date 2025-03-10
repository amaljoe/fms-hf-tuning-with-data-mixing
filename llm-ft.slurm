#!/bin/bash
#SBATCH --job-name=llm_finetune            # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks (processes)
#SBATCH --gres=gpu:1                       # Number of GPUs per node
#SBATCH --mem=32G                          # Total memory per node
#SBATCH --time=01:30:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                    # Partition name
#SBATCH --output=finetune_output.log       # Standard output log
#SBATCH --error=finetune_error.log         # Standard error log

# Load necessary modules
module load mldl_modules/miniconda

conda init
source ~/.bashrc
conda activate base

# Navigate to project directory
cd /home/IITB/cfilt/24m0797/fms-hf-tuning-with-data-mixing

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv -l 1 > gpu_metrics.csv &

# Run fine-tuning script
./ft.sh

pkill -f "nvidia-smi"