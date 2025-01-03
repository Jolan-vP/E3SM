#!/bin/bash
#SBATCH --job-name=exp014
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 03:30:00
#SBATCH -A m4620

module load python
conda activate env-torch

#run the application:
srun -n 128 -c 2 —cpu_bind=cores
/pscratch/sd/p/plutzner/E3SM/exp015.py

echo “PROGRAM COMPLETE”
