#!/bin/bash
#
#SBATCH --job-name=y8s-obb_20k_adam_coslr_500_64_3
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=y8n-obb_20k_adam_coslr_500_64_3.out
#SBATCH --error=y8n-obb_20k_adam_coslr_500_64_3.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sg7761@nyu.edu
module purge

# format y8n_20k_adam_coslr_300_64_0 - pretraiend model dataset optim scheduler epochs batch run
singularity exec --nv --overlay /scratch/sg7761/robomaster/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh;conda activate yolov8;cd /scratch/sg7761/robomaster; yolo detect train data=datasets/obb/twentyk/data.yaml model=yolov8s-obb.pt epochs=500 batch=64 plots=True name=y8s-obb_20k_adam_coslr_500_64_3 optimizer='Adam' cos_lr=True profile=True exist_ok=True simplify=True dynamic=True format='engine' half=True flipud=0.5 copy_paste=0.1"
