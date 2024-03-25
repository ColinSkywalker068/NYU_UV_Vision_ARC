#!/bin/bash
#
#SBATCH --job-name=nas_20k_adamw_coslr_150_32_0_qat
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=nas_20k_adamw_coslr_150_32_0_qat.out
#SBATCH --error=nas_20k_adamw_coslr_150_32_0_qat.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sg7761@nyu.edu
module purge

singularity exec --nv --overlay /scratch/sg7761/robomaster/yolonas/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh;conda activate yolo-nas;cd /scratch/sg7761/robomaster/yolonas/YOLO-NAS; python3 qat.py --data data.yaml --batch 32 --epoch 150 --model yolo_nas_s --size 640 --weight runs/nas_20k_adamw_cos_150_64_10/ckpt_best.pth"
