#!/bin/bash
#SBATCH --job-name=exp028_Training
#SBATCH -N 4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 02:30:00
#SBATCH -A m4620

module load python
conda activate env-torch
cd /pscratch/sd/p/plutzner/E3SM

#run the application:
srun SANDBOX.py

echo “PROGRAM COMPLETE”
