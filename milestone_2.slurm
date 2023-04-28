#!/bin/bash
#SBATCH --mail-user=cameron.rondeau@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --mem=15GB
#SBATCH --output=bigdata_project/myjob.out
#SBATCH --error=bigdata_project/myjob.err

module restore bigdata
source activate bigdata
~/.conda/envs/bigdata/bin/python /home/rondeac/bigdata_project/final_project_milestone2.py
