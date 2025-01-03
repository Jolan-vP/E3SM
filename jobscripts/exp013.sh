#!/bin/bash
#SBATCH --job-name=exp013TRAIN
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 05:00:00
#SBATCH -A m4620
#SBATCH --mem=64G

module load python
conda activate env-torch
cd /pscratch/sd/p/plutzner/E3SM 

#run the application:
srun -n 2 -c 256 --cpu_bind=cores python -u /pscratch/sd/p/plutzner/E3SM/exp013.py

echo “PROGRAM COMPLETE”
