#!/bin/bash
#
#SBATCH --job-name=y8s_20k_adam_coslr_300_64_1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=y8s_20k_adam_coslr_500_64_1.out
#SBATCH --error=y8s_20k_adam_coslr_500_64_1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sg7761@nyu.edu
module purge

# format y8n_20k_adam_coslr_300_64_0 - pretraiend model dataset optim scheduler epochs batch run
singularity exec --nv --overlay /scratch/sg7761/robomaster/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh;conda activate yolov8;cd /scratch/sg7761/robomaster; yolo detect train data=datasets/twentyk/data.yaml model=yolov8s.pt epochs=500 batch=64 plots=True name=y8s_20k_adam_coslr_500_64_0 optimizer='Adam' cos_lr=True profile=True exist_ok=True simplify=True dynamic=True format='engine' half=True flipud=0.5 copy_paste=0.1"