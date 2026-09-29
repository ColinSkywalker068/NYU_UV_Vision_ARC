#!/usr/bin/env python3
"""Bake the checksum-pinned Basler layer into an existing JetPack GA image."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

REPO = "nyuultraviolet/cv-dockerfile"
DIGEST = "e119ef30b9e1cc4510d2e9b4ff292d92e90750081466afa9350ff34731f6eaf7"


def request(url, headers=None):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers or {}), timeout=60
    )


def allowed(name):
    return (name == "opt/pylon" or name.startswith("opt/pylon/") or
            (name.startswith("usr/lib/") and "pylon" in PurePosixPath(name).name.lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded layer")
    parser.add_argument("--base-image", default=REPO + ":orin2-ds70-base")
    parser.add_argument("--target-image", default="cv-detection-orin2:jp60-ds70-pylon")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cv-pylon-") as directory:
        context = Path(directory)
        archive = args.archive or context / "sdk-layer.tar.gz"
        if not args.archive:
            with request("https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + REPO + ":pull") as response:
                token = json.load(response)["token"]
            with request("https://registry-1.docker.io/v2/" + REPO + "/blobs/sha256:" + DIGEST,
                         {"Authorization": "Bearer " + token}) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
        print("Verifying pinned Basler SDK layer checksum", flush=True)
        checksum = hashlib.sha256()
        with archive.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != DIGEST:
            raise RuntimeError("Basler layer checksum mismatch")
        print("Checksum verified; extracting Basler SDK and plugin", flush=True)
        rootfs = context / "rootfs"
        rootfs.mkdir()
        with tarfile.open(archive) as layer:
            members = []
            for member in layer:
                name = member.name.rstrip("/")
                if member.isdir() and name in ("opt", "usr", "usr/lib", "usr/lib/aarch64-linux-gnu", "usr/lib/aarch64-linux-gnu/gstreamer-1.0"):
                    continue
                if not allowed(name) or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                    raise RuntimeError("Unexpected SDK layer path: " + name)
                if member.issym() or member.islnk():
                    target = member.linkname
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), target)) if member.issym() else target
                    if not allowed(target) or PurePosixPath(target).is_absolute():
                        raise RuntimeError("Unsafe SDK layer link: " + name)
                elif not (member.isfile() or member.isdir()):
                    raise RuntimeError("Unexpected SDK layer entry: " + name)
                members.append(member)
            layer.extractall(rootfs, members=members)
        if not (rootfs / "opt/pylon/include/pylon/PylonIncludes.h").is_file():
            raise RuntimeError("Layer is missing the Basler C++ SDK")
        (context / "Dockerfile").write_text("""ARG BASE_IMAGE=nyuultraviolet/cv-dockerfile:orin2-ds70-base
FROM ${BASE_IMAGE}
COPY rootfs/ /
ENV PYLON_ROOT=/opt/pylon
ENV PATH=${PATH}:/opt/pylon/bin
RUN ldconfig && rm -rf /root/.cache/gstreamer-1.0
CMD ["/bin/bash"]
""")
        print("Building camera runtime with networking disabled", flush=True)
        subprocess.run(["docker", "build", "--network", "none", "--build-arg", "BASE_IMAGE=" + args.base_image,
                        "-t", args.target_image, str(context)], check=True)
        print("Checking pylonsrc without camera/serial devices", flush=True)
        subprocess.run(["docker", "run", "--rm", "--runtime", "nvidia", "--network", "none",
                        "--entrypoint", "gst-inspect-1.0", args.target_image, "pylonsrc"], check=True)


if __name__ == "__main__":
    main()
