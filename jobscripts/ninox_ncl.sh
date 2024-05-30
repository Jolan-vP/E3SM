#!/bin/bash
#SBATCH --job-name=ninox_NCL
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -c 2
#SBATCH -n 128
#SBATCH -q regular
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:15:00
#SBATCH -A m4620

module load e4s
spack env activate gcc
spack load --only package ncl
ncl /pscratch/sd/p/plutzner/E3SM/bigdata/E3SMv2data/ninox.ncl
