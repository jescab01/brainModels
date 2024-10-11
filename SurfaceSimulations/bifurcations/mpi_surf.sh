#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=short
#SBATCH --job-name=surf
#SBATCH --ntasks=50
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jescab01@ucm.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

module purge && module load Python/3.9.6-GCCcore-11.2.0

srun python mpi_surf.py


