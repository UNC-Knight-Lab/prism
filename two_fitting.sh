#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 4g
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=suprajac@email.unc.edu

module purge
module load anaconda
conda activate prism
python two_fitting.py