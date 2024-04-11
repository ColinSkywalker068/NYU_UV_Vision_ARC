# Weights and Biases

This readme is a quick quide to using wandb.ai to track your training experiements. With multiple people running training of the same model, its hard to go through all parameters and decide which model is better. Make sure you are added to the organization on https://wandb.ai/robomaster-nyu/

Here we see how to use the wandb.ai platform

<p align="center">
    <img src="../docs/img/wandb-demo.png" alt="wandb platform"> 
  <h5 align="center">
    Preview of the platform showing different runs
  </h5>
</p>


## Pre reqs
### Environment
You will have to run commands again from `Setup overlay image` section in [README.md](README.md).
Replace the first overlay path with your own. Make sure `rw` is enabled for the path as shown in the commands and activate your environment.

**NOTE** `tensorrt` and `onnx` modules are completely optional. I use it for profiling as we will be deploying the model in those formats. Its recommended but not required for training. `wandb` is the most important change to your environment here.

```
srun --pty -c 4 --mem=16GB --time=01:30:00 --gres=gpu:rtx8000:1 /bin/bash
singularity exec --nv --overlay /scratch/sg7761/robomaster/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
source /ext3/env.sh
conda activate yolov8
```

**Option 1**

Refer to `yml/environment_yolov8.yml` for all the new dependencies. 
```
conda env update --file yml/environment_yolov8.yml --prune
```

**Option 2**

Here are the commands I ran manually on the exisiting `yolov8` environment. 
```
python3 -m pip install --pre --upgrade tensorrt
pip3 install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs
pip3 install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt_bindings
pip3 install —no-cached-dir onnx onnxsim onnxruntime-gpu onnxruntime numpy==1.23.1 wandb
```
reference - [Tensorrt installation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)

**NOTE** None of the above installations should occur in `/home/<netid>/.local/` or `/home/<netid>/.cache/`. Do not use `home` folder to install dependencies

## Testing
Launch the Python3 interpreter
```
import tensorrt as trt
trt.__version__
import wandb
```
Your `tensorrt` version should be `8.6.1` if youre running cuda-12. `wandb` should import without any complains.

## Wandb setup
Continuing in the `srun` environment and your `yolob8` environment activated, run the following commands:
```
wandb login
```
Follow the prompt instructions and you should be connected now.

## Training
**NOTE** The guide below also shows how to install `tensorrt` which is the model format we use when deploying our model on the jetson. We use it for profiling purposes but you can also just remove the profiling flags such has `profile`, `half`, `format`

A new `sbatch` script has been shared `train_ultralytics.SBATCH` which runs `train_ultralytics.py` internally. Most `ultralytics` have been wrapped. In case you want additional flags, feel free to modify the python file and make a PR.

**NOTE** Wandb by default caches logs in `/home/<netdi>/.cache/wandb`, to modify this, edit the environment variable set in `train_ultralytics.py` to either use `/scratch/<netdi>` or `/vast/<netdi>`

After altering the `sbatch` file to your folder paths and environment, you can launch using
```
sbatch train_ultralytics.SBATCH
``` 

You should be able to monitor your logs here once resources are allocated - https://wandb.ai/robomaster-nyu/comp24-models

You can now run multiple experiments and share interesting results with the team!