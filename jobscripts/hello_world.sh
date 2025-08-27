#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -A m4620
#SBATCH --mail-user=j.vonplutzner@colostate.edu
#SBATCH --mail-type=ALL

echo "Hello World" 