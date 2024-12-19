#!/bin/bash
#SBATCH --job-name=gtraj             # Job name
#SBATCH --mem=60000                     # Job memory request
#SBATCH -t 2-23:59               # Time limit days-hrs:min:sec
#SBATCH -N 1                         # requested number of nodes (usually just 1)
#SBATCH -n 10                       # requested number of CPUs
#SBATCH -p scavenger             # requested partition on which the job will run
#SBATCH --output=outputs/gait/trajectories.out   # file path to slurm output
# SBATCH --array=0-9                 # job array

# python work.py $SLURM_ARRAY_TASK_ID
python work.py 2

