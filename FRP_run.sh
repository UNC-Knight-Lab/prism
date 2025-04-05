#!/bin/bash

#SBATCH --ntasks=8
#SBATCH -N 1
#SBATCH --mem 16000MB
#SBATCH -t 48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=suprajac@email.unc.edu

module purge
module load anaconda
conda activate prism
python -u FRP.py