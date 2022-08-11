#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard
#SBATCH --job-name=dynSys_s
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=5G
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=F_r_e@hotmail.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

module purge && module load Python

srun python dynSysPSE_mpi_s.py


