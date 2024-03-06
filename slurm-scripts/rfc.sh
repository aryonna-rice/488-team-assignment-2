#!/bin/bash
#SBATCH --partition=general
#SBATCH --time=2:00:00  # Adjust based on your estimation of job completion time
#SBATCH --mem=128G  # Adjust based on the node selection and your job's memory requirement
#SBATCH --cpus-per-task=48  # Adjust to leverage parallel computation
#SBATCH --mail-type=all
#SBATCH --mail-user=aryonna@email.unc.edu

# Activate your virtual environment
source /nas/longleaf/home/aryonna/488-team-assignment-2/488env/bin/activate

sar -u -r 60 > cpu_mem_usage_$SLURM_JOB_ID.txt &
# Run your Python script
python /nas/longleaf/home/aryonna/488-team-assignment-2/training-scripts/rfc.py

wait

echo "Job ended at `date`"