#!/bin/bash
#
#SBATCH --job-name=yolov8s-v13-adam-coslr-500-64-0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=./slurm/yolov8s-v13-adam-coslr-500-64-0.out
#SBATCH --error=./slurm/yolov8s-v13-adam-coslr-500-64-0.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sg7761@nyu.edu
module purge

# format y8n_20k_adam_coslr_300_64_0 - pretraiend model dataset optim scheduler epochs batch run
# MAKE SURE TO CREATE A `slurm` FOLDER FOR OUTPUTS BEFORE RUNNING THIS FILE

singularity exec --nv --overlay /scratch/sg7761/robomaster/overlay-25GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "mkdir -p /scratch/sg7761/.cache/wandb; WANDB_CACHE_DIR=/scratch/sg7761/.cache/wandb; \
source /ext3/env.sh; conda activate yolov8;cd /scratch/sg7761/robomaster; \
python3 train_ultralytics.py --data /scratch/sg7761/robomaster/datasets/v13/data.yaml \
--model yolov8s.pt --optimizer Adam --epochs 500 --cos_lr --plots --profile  --format engine --half \
--flipud 0.5 --copy_paste 0.2 --mixup 0.2 --name yolov8s-v13-adam-coslr-500-64-0"