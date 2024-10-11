#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=long
#SBATCH --job-name=D5pse
#SBATCH --ntasks=40
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jescab01@ucm.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

srun python mpi_D5pse.py


