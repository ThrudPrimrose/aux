#!/bin/bash
#SBATCH -J ex
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --ntasks-per-node=1
# 56 is the maximum reasonable value for CooLMUC-2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yakup.paradox@gmail.com
#SBATCH --export=NONE
#SBATCH --time=00:00:59

module load slurm_setup
module load python/3.8.11-extended

echo "Hello"
python ex.py