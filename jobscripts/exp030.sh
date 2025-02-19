#!/bin/bash
#SBATCH --job-name=exp030_Training
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 8:15:00
#SBATCH -A m4620

cd /pscratch/sd/p/plutzner/E3SM
module load python
conda activate env-torch

echo “STARTING PROGRAM”

#run the application:
srun -n 1 -c 256 -u --cpu-bind=cores python -u SANDBOX.py

echo “PROGRAM COMPLETE”
