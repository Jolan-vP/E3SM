#!/bin/bash
#SBATCH --job-name=ninox
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:25:00
#SBATCH -A m4620

module load conda
conda activate env-torch

#run the application:
srun -n 128 -c 2 —cpu_bind=cores
/pscratch/sd/p/plutzner/E3SM/databuilder/nino_indices.py

echo “PROGRAM COMPLETE”
