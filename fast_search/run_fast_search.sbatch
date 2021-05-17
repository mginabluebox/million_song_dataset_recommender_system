#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --array=1
#SBATCH --job-name=fast_search
#SBATCH --output=slurm_output/slurm_%A_%a.out

module purge

singularity exec --nv \
        --overlay /scratch/tj810/final-project-team_unsupervised_learners/overlay-5GB-200K.ext3:ro \
        /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
        bash -c "source /ext3/env.sh;\
                 python fast_search.py $SLURM_ARRAY_TASK_ID True;"
