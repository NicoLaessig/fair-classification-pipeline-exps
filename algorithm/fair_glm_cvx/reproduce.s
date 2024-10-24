#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=FGLM
#SBATCH --output=./logs/FGLM_%a.out
#SBATCH -a 0-11

module load gurobi/9.1.1

datasets=("adult" "arrhythmia" "compas" "crime" "drug_consumption" "drug_consumption_multi" "german_credit" "hrs" "lsac" "obesity" "parkinsons_updrs" "student_performance")
dataset="${datasets[SLURM_ARRAY_TASK_ID]}"

echo $SLURM_ARRAY_TASK_ID, $dataset

/share/apps/singularity/bin/singularity \
    exec \
    --overlay $SCRATCH/containers/fair-glm.ext3:ro \
    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
    /bin/bash -c "
source /ext3/env.sh
python experiment.py --dataset $dataset
"