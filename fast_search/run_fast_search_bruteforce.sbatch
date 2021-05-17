#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=128GB
#SBATCH --array=1,50-500:50
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=tj810@nyu.edu
#SBATCH --job-name=fast_search_bruteforce
#SBATCH --output=slurm_output/slurm_%A_%a.out
#SBATCH --error=slurm_output/slurm_%A_%a.err

module purge

singularity exec --overlay /scratch/tj810/final-project-team_unsupervised_learners/overlay-5GB-200K.ext3:ro \
        /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
        bash -c "source /ext3/env.sh;\
                 python fast_search.py $SLURM_ARRAY_TASK_ID bruteforce;"
