#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=short
#SBATCH --job-name=bnm-param
#SBATCH --ntasks=150
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jescab01@ucm.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

srun python mpi_bnm.py


