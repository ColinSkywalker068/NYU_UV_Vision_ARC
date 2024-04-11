"""
make sure data is in right format, i am training yolov8 so i downloaded in yolov8 format
yolov8 and yolov8-obb are two different formats!

python3 train_ultralytics.py --data /scratch/sg7761/robomaster/datasets/triton/data.yaml \
--model yolov8n.pt --optimizer Adam --cos_lr --plots --profile --simplify --dynamic \
--format engine --half --flipud 0.5 --copy_paste 0.1 --mixup 0.1
"""

import argparse
from ultralytics import YOLO, RTDETR
from wandb.integration.ultralytics import add_wandb_callback
import wandb
import os

os.environ["WANDB_CACHE_DIR"] = "/scratch/sg7761/.cache/wandb"

# Setup argument parser
parser = argparse.ArgumentParser(description="Train YOLOv8 Model with W&B Logging")
parser.add_argument(
    "--data", type=str, default="datasets/twentyk/data.yaml", help="Dataset YAML path"
)
parser.add_argument("--model", type=str, default="yolov8s.pt", help="Model filename")
parser.add_argument(
    "--task",
    type=str,
    default="detect",
    help="detect for all/ obb is supported only by yolov8",
)
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--batch", type=int, default=64, help="Batch size")
parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer")
parser.add_argument("--cos_lr", action="store_true", help="Use cosine LR schedule")
parser.add_argument("--name", type=str, default="", help="Run name")
parser.add_argument("--plots", action="store_true", help="Plot training results")
parser.add_argument("--profile", action="store_true", help="Use profiler")
parser.add_argument("--exist_ok", action="store_true", help="Existing project is okay")
parser.add_argument("--simplify", action="store_true", help="Simplify onnx")
parser.add_argument("--dynamic", action="store_true", help="Use dynamic training")
parser.add_argument("--format", type=str, default="engine", help="Output format")
parser.add_argument("--half", action="store_true", help="Use FP16 precision")
parser.add_argument(
    "--flipud", type=float, default=0.0, help="Flip image up-down probability"
)
parser.add_argument(
    "--copy_paste", type=float, default=0.0, help="Copy-Paste augmentation probability"
)
parser.add_argument("--mixup", type=float, default=0.0, help="mixup probability")

args = parser.parse_args()

# Initialize a Weights & Biases run
wandb.init(
    project="comp24-models", job_type="training", config=args, entity="robomaster-nyu"
)

# Define the YOLOv8 model and dataset using argparse parameters
if args.model[:4] == "yolo":
    model = YOLO(args.model, args.task)
else:
    model = RTDETR(args.model)

# Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train and fine-tune the model
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=640,
    batch=args.batch,
    name=args.name,
    optimizer=args.optimizer,
    cos_lr=args.cos_lr,
    plots=args.plots,
    profile=args.profile,
    exist_ok=args.exist_ok,
    simplify=args.simplify,
    dynamic=args.dynamic,
    format=args.format,
    half=args.half,
    flipud=args.flipud,
    copy_paste=args.copy_paste,
    mixup=args.mixup,
)

wandb.finish()
