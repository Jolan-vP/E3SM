#!/bin/bash
#SBATCH --job-name=exp016PROCESSING
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 03:30:00
#SBATCH -A m4620

module load python
conda activate env-torch
cd /pscratch/sd/p/plutzner/E3SM

#run the application:
srun -n 2 -c 128 —cpu_bind=cores python -u /pscratch/sd/p/plutzner/E3SM/exp016.py

echo “PROGRAM COMPLETE”
