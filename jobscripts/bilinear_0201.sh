#!/bin/bash
#SBATCH --job-name=bil0201
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:05:00
#SBATCH -A m4620

source
/global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.csh

#run the application:
srun -n 128 -c 2 —cpu_bind=cores
/pscratch/sd/p/plutzner/E3SM/E3SMv2data/member0201/bilinear_interpolation_0201.csh

echo “PROGRAM COMPLETE”
