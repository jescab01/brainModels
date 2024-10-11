#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=long
#SBATCH --job-name=hierarchies
#SBATCH --ntasks=75
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jescab01@ucm.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

srun python mpi_hierarchies.py


