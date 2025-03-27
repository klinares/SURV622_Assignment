#!/bin/bash
#SBATCH --nodes=1          # Use 1 Node     (Unless code is multi-node parallelized)
#SBATCH --ntasks=1
#SBATCH --account=<your_account>
#SBATCH --time=23:20:00
#SBATCH --cpus-per-task=3
#SBATCH -o slurm-%j.out-%N
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=180000m
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>   # Your email address has to be set accordingly
#SBATCH --job-name=<name_your_project>        # the job's name you want to be used

module load python3.10-anaconda

export FILENAME=
srun python FILENAME > SLURM_JOBID.out

echo "End of program at `date`"