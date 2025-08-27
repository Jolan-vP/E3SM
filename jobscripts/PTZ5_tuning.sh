#!/bin/bash
#SBATCH --job-name=PTZ5_tuning
#SBATCH --array=133-139
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/config_%A_%a.out
#SBATCH --error=logs/config_%A_%a.err

# Path to configuration file
CONFIG_FILE="configs/config_${SLURM_ARRAY_TASK_ID}.json"

echo "Using configuration file: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Run training with configuration file
python SANDBOX.py --config $CONFIG_FILE
