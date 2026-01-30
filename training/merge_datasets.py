import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Union, Set

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths to the data.yaml files of the two datasets to merge
DATASET_A_YAML = "/home/jupyter-robomaster/KushalNew/random-2/data.yaml"
DATASET_B_YAML = "/home/jupyter-robomaster/KushalNew/RoboMaster-1/data.yaml"

# Output directory for the merged dataset
OUTPUT_DIR = "mergeRM"

# Classes to keep from each dataset.
# You can use class names (strings) or class IDs (integers).
# If using names, they must match exactly what is in the data.yaml 'names' list.
# These will be mapped to a single class (ID 0) in the output.
DATASET_A_KEEP = ["blue-armor", "red-armor"]  # Example: Keep 'person' and 'car' from A
DATASET_B_KEEP = [0, 2]  # Example: Keep class ID 0 and 2 from B

# The name of the single class in the new dataset
TARGET_CLASS_NAME = "object"

# ==============================================================================
# SCRIPT
# ==============================================================================


def load_yaml(path: Union[str, Path]) -> dict:
    """Loads a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: Union[str, Path]):
    """Creates a directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_keep_indices(yaml_data: dict, keep_list: list[Union[str, int]]) -> set[int]:
    """
    Converts a mixed list of class names and IDs into a set of class IDs
    valid for the given dataset YAML.
    """
    names = yaml_data.get("names", {})
    # Handle 'names' being a list or a dict
    if isinstance(names, list):
        name_to_id = {name: i for i, name in enumerate(names)}
    elif isinstance(names, dict):
        name_to_id = {val: key for key, val in names.items()}
    else:
        raise ValueError(f"Unknown format for 'names' in YAML: {type(names)}")

    keep_ids = set()
    for item in keep_list:
        if isinstance(item, int):
            keep_ids.add(item)
        elif isinstance(item, str):
            if item in name_to_id:
                keep_ids.add(name_to_id[item])
            else:
                print(f"Warning: Class '{item}' not found in dataset names. Skipping.")
        else:
            print(f"Warning: Invalid type for keep item '{item}'. Skipping.")

    return keep_ids


def convert_xywh_to_obb(xc, yc, w, h):
    """
    Converts normalized xywh (center-based) to normalized OBB (4 corners).
    Returns list of 8 floats: [x1, y1, x2, y2, x3, y3, x4, y4]
    Points are ordered: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc - h / 2
    x3 = xc + w / 2
    y3 = yc + h / 2
    x4 = xc - w / 2
    y4 = yc + h / 2
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def process_dataset(
    dataset_yaml_path: Path, output_root: Path, keep_ids: set[int], prefix: str
):
    """
    Processes a single dataset: copies images and filters/remaps labels.
    Converts standard bounding boxes (xywh) to OBB (x1y1...x4y4).
    """
    dataset_root = dataset_yaml_path.parent
    data_config = load_yaml(dataset_yaml_path)

    # YOLOv8 standard structure usually expects keys like 'train', 'val', 'test'
    # pointing to image directories relative to the yaml or absolute.

    splits = ["train", "val", "test"]

    for split in splits:
        if split not in data_config:
            continue

        rel_path = data_config[split]
        # Handle cases where path might be absolute or relative
        img_src_dir = (
            (dataset_root / rel_path).resolve()
            if not Path(rel_path).is_absolute()
            else Path(rel_path)
        )

        if not img_src_dir.exists():
            print(
                f"Warning: Image directory {img_src_dir} does not exist. Skipping split {split}."
            )
            continue

        # Prepare output directories
        out_img_dir = output_root / "images" / split
        out_lbl_dir = output_root / "labels" / split
        ensure_dir(out_img_dir)
        ensure_dir(out_lbl_dir)

        # Find images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [
            p for p in img_src_dir.iterdir() if p.suffix.lower() in valid_extensions
        ]

        print(f"Processing {split} set for {prefix} ({len(images)} images)...")

        for img_path in tqdm(images):
            # Define new filenames with prefix to avoid collisions
            new_filename = f"{prefix}_{img_path.name}"
            out_img_path = out_img_dir / new_filename

            # Copy image
            shutil.copy2(img_path, out_img_path)

            # Process label
            # Assuming label is in ../labels/{split}/image_name.txt relative to image dir
            # OR standard YOLO layout: images/train -> labels/train

            # Try to find corresponding label file
            # Strategy 1: Replace 'images' with 'labels' in path
            parts = list(img_path.parts)
            try:
                # Find the right-most 'images' occurrence to replace
                idx = len(parts) - 1 - parts[::-1].index("images")
                parts[idx] = "labels"
                lbl_path = Path(*parts).with_suffix(".txt")
            except ValueError:
                # 'images' not in path, try simple sibling assumption
                lbl_path = (
                    img_path.parent.parent
                    / "labels"
                    / split
                    / img_path.with_suffix(".txt").name
                )

            if not lbl_path.exists():
                # If no label file, just skip creating a label file (background image)
                continue

            # Filter, remap, and convert labels to OBB
            new_lines = []
            with open(lbl_path) as f:
                for line in f:
                    parts_line = line.strip().split()
                    if not parts_line:
                        continue

                    try:
                        class_id = int(parts_line[0])
                        # Check if we keep this class
                        if class_id in keep_ids:
                            # Parse coordinates
                            coords = [float(x) for x in parts_line[1:]]

                            coord_str = ""
                            if len(coords) == 4:
                                # Standard xywh -> Convert to OBB
                                obb_coords = convert_xywh_to_obb(*coords)
                                coord_str = " ".join(f"{x:.6f}" for x in obb_coords)
                            elif len(coords) >= 8:
                                # Already OBB (x1, y1, x2, y2, x3, y3, x4, y4)
                                # Take first 8
                                obb_coords = coords[:8]
                                coord_str = " ".join(f"{x:.6f}" for x in obb_coords)
                            else:
                                # Unknown format, skip
                                # print(f"Warning: Skipped line with {len(coords)} coords in {lbl_path}")
                                continue

                            # Remap to class 0
                            new_line = f"0 {coord_str}\n"
                            new_lines.append(new_line)
                    except ValueError:
                        continue

            # Write new label file if we have valid annotations
            if new_lines:
                out_lbl_path = out_lbl_dir / f"{prefix}_{img_path.stem}.txt"
                with open(out_lbl_path, "w") as f:
                    f.writelines(new_lines)


def merge_datasets():
    """Main function to control the merge process."""
    print("Starting dataset merge (OBB mode)...")

    path_a = Path(DATASET_A_YAML)
    path_b = Path(DATASET_B_YAML)
    out_path = Path(OUTPUT_DIR)

    if not path_a.exists():
        print(f"Error: Dataset A YAML not found at {path_a}")
        return
    if not path_b.exists():
        print(f"Error: Dataset B YAML not found at {path_b}")
        return

    # Create output root
    if out_path.exists():
        resp = input(f"Output directory {out_path} exists. Overwrite? (y/n): ")
        if resp.lower() != "y":
            print("Aborted.")
            return
        shutil.rmtree(out_path)
    ensure_dir(out_path)

    # Load configs to resolve class names
    config_a = load_yaml(path_a)
    config_b = load_yaml(path_b)

    # Resolve class indices
    keep_ids_a = resolve_keep_indices(config_a, DATASET_A_KEEP)
    keep_ids_b = resolve_keep_indices(config_b, DATASET_B_KEEP)

    print(f"Keeping IDs from A: {keep_ids_a}")
    print(f"Keeping IDs from B: {keep_ids_b}")

    # Process Dataset A
    process_dataset(path_a, out_path, keep_ids_a, prefix="ds1")

    # Process Dataset B
    process_dataset(path_b, out_path, keep_ids_b, prefix="ds2")

    # Generate new data.yaml
    new_yaml = {
        "path": str(out_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",  # Optional, but including for completeness
        "nc": 1,
        "names": [TARGET_CLASS_NAME],
        "task": "obb",  # Explicitly mark as OBB
    }

    with open(out_path / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f, sort_keys=False)

    print(f"\nMerge complete! New OBB dataset saved to: {out_path}")
    print(f"New data.yaml created at: {out_path / 'data.yaml'}")


if __name__ == "__main__":
    merge_datasets()
