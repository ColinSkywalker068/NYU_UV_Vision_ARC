#!/usr/bin/env python3
"""Preserve ONNX detector metadata on a trtexec-built TensorRT plan.

Requires python3-onnx. Build the plan on the target Jetson/SDK first; this tool
does not convert engines across TensorRT versions or access robot hardware.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct


def main():
    import onnx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.plan.resolve(), args.onnx.resolve()):
        parser.error('Output must be separate from the source ONNX and raw plan')
    model = onnx.load(args.onnx)
    metadata = {item.key: item.value for item in model.metadata_props}
    for key in ('names', 'imgsz', 'args', 'end2end'):
        if key in metadata:
            metadata[key] = ast.literal_eval(metadata[key])
    if metadata.get('task') not in ('obb', 'detect') or not metadata.get('names'):
        parser.error('Source model must declare its detector task and class names')
    metadata['source_onnx_sha256'] = hashlib.sha256(args.onnx.read_bytes()).hexdigest()
    encoded = json.dumps(metadata).encode('utf-8')
    plan = args.plan.read_bytes()
    if not plan:
        parser.error('Raw TensorRT plan is empty')
    args.output.write_bytes(struct.pack('<I', len(encoded)) + encoded + plan)
    print(f'Wrote {args.output}: task={metadata["task"]}, names={metadata["names"]}')


if __name__ == '__main__':
    main()
