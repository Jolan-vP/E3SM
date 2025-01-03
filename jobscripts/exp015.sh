#!/bin/bash
#SBATCH --job-name=exp015PROCESSING
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00
#SBATCH -A m4620
#SBATCH --mem=64G

module load python
conda activate env-torch
cd /pscratch/sd/p/plutzner/E3SM 

#run the application:
srun -n 1 -c 256 --cpu_bind=cores python -u /pscratch/sd/p/plutzner/E3SM/exp015.py

echo “PROGRAM COMPLETE”
