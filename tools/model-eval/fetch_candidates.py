"""Download pinned public model files and inspect metadata without executing them."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "downloads"

def github_item(repo, revision, remote_path, name, size, blob):
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/" + urllib.parse.quote(remote_path)
    return dict(name=name, repo=repo, revision=revision, remote_path=remote_path,
                url=url, size=size, expected_git_blob_sha1=blob)

ITEMS = [
    dict(name="aruw-2026/5_12_icon_detector.onnx", repo="aruw/vision/project-otto",
         revision="154035fc34ffb34f99120ffa8e8a2bb098bc6306",
         remote_path="assets/models/5_12_icon_detector.onnx",
         url="https://gitlab.com/api/v4/projects/28205715/repository/files/assets%2Fmodels%2F5_12_icon_detector.onnx/raw?ref=154035fc34ffb34f99120ffa8e8a2bb098bc6306",
         size=10111317, expected_git_blob_sha1="0a2c97a0074c758f27afb5b7635cf7e59a21da86",
         expected_sha256="3af4e44cc66d0f2a29cb7912c9c8da4cdb902fccbc02df4f5b3c0d5847dcdefe"),
    github_item("SZURPVision/26_NNDeployment_Lib_and_Detection_Models",
                "ccb5dfe09696edeea445ffb8f99adaa7c383ec3a",
                "所有模型/onnx/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx",
                "robotpilots-2026/Infantry-v8n-fp16.onnx", 6429469,
                "aec315f1ea8aba20f8faaf269a19b881d7514242"),
    github_item("TongjiSuperPower/sp_vision_25", "bd9f5e798fa3c6dd3b483ae6627796afb41c608d",
                "assets/yolov5.xml", "tongji-2025/yolov5.xml", 349369,
                "88b2f682f19dd232172fd8f3b8f5b4e1423bf8e5"),
    github_item("TongjiSuperPower/sp_vision_25", "bd9f5e798fa3c6dd3b483ae6627796afb41c608d",
                "assets/yolov5.bin", "tongji-2025/yolov5.bin", 2836140,
                "e5f842740d73ceb9cbd0e21d63f2f166ae175014"),
    github_item("tup-robomaster/TUP-Vision-2023-Based", "40b8269ef8b0b9c3475911a06d6cd13a6e1257f2",
                "src/vehicle_system/autoaim/armor_detector/model/opt-1208-001.onnx",
                "tup-2023/opt-1208-001.onnx", 2167983,
                "6a1916c984e729b68da358b38085816893339109"),
]

def varint(data, offset):
    value = shift = 0
    for _ in range(10):
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7
    raise ValueError("Invalid protobuf varint")

def fields(data):
    """Read wire fields only; field numbers follow the official ONNX schema."""
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 2:
            count, offset = varint(data, offset)
            value = data[offset:offset+count]
            offset += count
        elif wire in (1, 5):
            count = 8 if wire == 1 else 4
            value = data[offset:offset+count]
            offset += count
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        if offset > len(data):
            raise ValueError("Truncated protobuf")
        yield number, value

def tensor_info(value):
    value = dict(fields(value))
    tensor = dict(fields(dict(fields(value[2]))[1]))
    shape = []
    for number, dimension in fields(tensor.get(2, b"")):
        if number == 1:
            d = dict(fields(dimension))
            shape.append(d.get(1, d.get(2, b"?").decode()))
    return dict(name=value[1].decode(), element_type=tensor[1], shape=shape)

def inspect_onnx(data):
    model = dict(fields(data))
    result = dict(ir_version=model.get(1), inputs=[], outputs=[])
    result["metadata"] = {}
    for number, value in fields(data):
        if number == 14:
            entry = dict(fields(value))
            result["metadata"][entry[1].decode()] = entry[2].decode()
    for number, value in fields(model[7]):
        if number in (11, 12):
            result["inputs" if number == 11 else "outputs"].append(tensor_info(value))
    return result

def fetch(item):
    path = DEST / item["name"]
    data = path.read_bytes() if path.exists() else urllib.request.urlopen(item["url"], timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if len(data) != item["size"]: raise ValueError(f"Size mismatch: {item['name']}")
    if blob != item["expected_git_blob_sha1"]: raise ValueError(f"Blob mismatch: {item['name']}")
    if "expected_sha256" in item and digest != item["expected_sha256"]: raise ValueError("SHA256 mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    result = dict(item, local_path=str(path), sha256=digest, verified=True)
    if path.suffix == ".onnx":
        result["onnx_metadata"] = inspect_onnx(data)
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for private evaluation weights and manifest")
    parser.add_argument("--candidate", choices=[i["name"].split("/")[0] for i in ITEMS])
    args = parser.parse_args()
    ROOT = args.output.resolve()
    DEST = ROOT / "downloads"
    ROOT.mkdir(parents=True, exist_ok=True)
    ITEMS = [i for i in ITEMS if not args.candidate or i["name"].startswith(args.candidate + "/")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch, ITEMS))
    (ROOT / "download-manifest.json").write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n")
    for item in results:
        summary = {k:item[k] for k in ("name", "size", "verified")}
        if "onnx_metadata" in item:
            summary["tensors"] = {k:item["onnx_metadata"][k] for k in ("inputs", "outputs")}
        print(json.dumps(summary))
