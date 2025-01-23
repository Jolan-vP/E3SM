#!/bin/bash
#SBATCH --job-name=exp015PROCESSING
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 3:30:0
#SBATCH -A m4620

module load python
conda activate env-torch
cd /pscratch/sd/p/plutzner/E3SM 

#run the application:
srun -n 2 -c 128 --cpu_bind=cores python -u /pscratch/sd/p/plutzner/E3SM/exp015.py

echo “PROGRAM COMPLETE”
