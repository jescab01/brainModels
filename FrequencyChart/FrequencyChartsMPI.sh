#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=short
#SBATCH --job-name=FreqCharts
#SBATCH --ntasks=100
#SBATCH --time=08:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=F_r_e@hotmail.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

srun python FrequencyChartsMPI.py


