#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --mail-type=ALL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=eleni.metheniti@irit.fr     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=00:05:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
axn7833; osirim-slurm.irit.fr; 18/11/2019

module load python

python /projets/melodi/lenakmeth/main.py --lang hun