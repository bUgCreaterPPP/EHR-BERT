#!/bin/bash

#SBATCH -A birthright
#SBATCH -p burst
#SBATCH -N 2
#SBATCH -c 4
#SBATCH --mem=300g
#SBATCH -t 05:00:00
#SBATCH -o ./output/output.%A.out
python PAC_demo.py -d $1 -th $2
