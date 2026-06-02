"""GPU / process preflight for YOLOv8 ROS nodes.

Avoids the heavy-handed ``fuser -k /dev/nvhost-*`` pattern, which happily
kills the desktop compositor, camera stack, and anything else touching the
Jetson's NVGPU device nodes. Instead we do three things:

1. ``check``  (default): report what's running and how much GPU memory is in
   use. Purely read-only so it's always safe to run from a launch file.
2. ``clean``: additionally send SIGTERM to *known* stale YOLO / camera /
   bridge processes left behind by previous runs of this package, so we can
   re-bind serial ports and GPU memory. Still targeted — never a blanket
   ``fuser -k``.
3. ``force``: clean + aggressive. Escalate surviving matches to SIGKILL and
   call ``torch.cuda.empty_cache()`` / ``torch.cuda.ipc_collect()`` in-
   process. Still does not touch arbitrary CUDA clients.

Dropping the page cache (``echo 3 > drop_caches``) is intentionally *not*
included: it does nothing for VRAM and usually hurts startup latency.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass

LOGGER = logging.getLogger("yolov8_preflight")

_DEFAULT_SUBPROCESS_TIMEOUT_S: float = 5.0
_SIGTERM_GRACE_PERIOD_S: float = 2.0

KNOWN_CONFLICTING_PROCESS_NAMES: tuple[str, ...] = (
    "camera_node",
    "yolov8_modular_node",
    "yolov8_node",
    "stm32_serial_bridge",
    "stm32_serial_bridge_node",
    "realsense2_camera_node",
)


@dataclass(frozen=True)
class PreflightOptions:
    mode: str = "check"
    verbose: bool = False
    current_pid: int = 0


@dataclass(frozen=True)
class PreflightReport:
    killed_pids: tuple[int, ...]
    survived_pids: tuple[int, ...]
    gpu_memory_used_mb: float
    cuda_cache_cleared: bool


def _hasCommand(name: str) -> bool:
    return shutil.which(name) is not None


def _runCommand(cmd: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_SUBPROCESS_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as err:
        LOGGER.debug("preflight command failed %s: %s", cmd, err)
        return 1, ""
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _findConflictingPids(current_pid: int) -> list[int]:
    """Return PIDs of stale YOLO / camera / bridge processes, excluding self."""
    if not _hasCommand("pgrep"):
        return []
    pids: set[int] = set()
    for name in KNOWN_CONFLICTING_PROCESS_NAMES:
        code, out = _runCommand(["pgrep", "-f", name])
        if code != 0:
            continue
        for line in out.splitlines():
            line = line.strip()
            if not line.isdigit():
                continue
            pid = int(line)
            if pid == current_pid or pid == os.getpid():
                continue
            pids.add(pid)
    return sorted(pids)


def _readGpuMemoryUsedMb() -> float:
    """Best-effort GPU memory in MB. Works on discrete NVIDIA + Jetson (tegrastats)."""
    if _hasCommand("nvidia-smi"):
        code, out = _runCommand(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        if code == 0 and out.strip():
            try:
                return float(out.strip().splitlines()[0])
            except ValueError:
                return 0.0
    if _hasCommand("tegrastats"):
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "100"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (FileNotFoundError, OSError):
            return 0.0
        try:
            line = proc.stdout.readline() if proc.stdout else ""
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        marker = "RAM "
        idx = line.find(marker)
        if idx < 0:
            return 0.0
        tail = line[idx + len(marker) :].split("/")[0].strip()
        try:
            return float(tail)
        except ValueError:
            return 0.0
    return 0.0


def _signalProcess(pid: int, sig: int) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as err:
        LOGGER.warning("Missing permission to signal pid=%d: %s", pid, err)
        return False


def _terminatePids(pids: list[int], force: bool) -> tuple[list[int], list[int]]:
    if not pids:
        return [], []
    for pid in pids:
        _signalProcess(pid, signal.SIGTERM)
    time.sleep(_SIGTERM_GRACE_PERIOD_S)
    survivors: list[int] = [pid for pid in pids if _signalProcess(pid, 0)]
    if survivors and force:
        for pid in survivors:
            _signalProcess(pid, signal.SIGKILL)
        time.sleep(0.5)
        survivors = [pid for pid in survivors if _signalProcess(pid, 0)]
    killed: list[int] = [pid for pid in pids if pid not in survivors]
    return killed, survivors


def _clearCudaCache() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        return True
    except Exception as err:
        LOGGER.debug("torch.cuda cleanup failed: %s", err)
        return False


def executePreflight(options: PreflightOptions) -> PreflightReport:
    """Run the configured preflight and return a structured report."""
    mode = (options.mode or "check").strip().lower()
    if mode not in ("check", "clean", "force"):
        LOGGER.warning("Unknown preflight mode '%s' — falling back to 'check'", mode)
        mode = "check"
    gpu_used = _readGpuMemoryUsedMb()
    pids = _findConflictingPids(options.current_pid)
    if options.verbose or pids:
        LOGGER.info(
            "preflight mode=%s gpu_used=%.1f MB stale_pids=%s",
            mode,
            gpu_used,
            pids or "[]",
        )
    if mode == "check" or not pids:
        cuda_cleared = False
        if mode == "force":
            cuda_cleared = _clearCudaCache()
        return PreflightReport(
            killed_pids=tuple(),
            survived_pids=tuple(pids),
            gpu_memory_used_mb=gpu_used,
            cuda_cache_cleared=cuda_cleared,
        )
    killed, survivors = _terminatePids(pids, force=(mode == "force"))
    if survivors:
        LOGGER.warning("preflight could not reclaim pids=%s (mode=%s)", survivors, mode)
    cuda_cleared = _clearCudaCache() if mode == "force" else False
    return PreflightReport(
        killed_pids=tuple(killed),
        survived_pids=tuple(survivors),
        gpu_memory_used_mb=gpu_used,
        cuda_cache_cleared=cuda_cleared,
    )


def runLaunchPreflight(mode: str, force_gpu_clean: bool, verbose: bool) -> None:
    """Convenience entry used from ``stm.launch.py``. Never raises."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="[preflight] %(message)s")
    effective_mode = "force" if force_gpu_clean else mode
    try:
        report = executePreflight(
            PreflightOptions(
                mode=effective_mode,
                verbose=verbose,
                current_pid=os.getpid(),
            )
        )
    except Exception as err:
        LOGGER.warning("Preflight skipped due to unexpected error: %s", err)
        return
    LOGGER.info(
        "Preflight done: mode=%s killed=%s survived=%s gpu_used=%.1f MB cuda_cleared=%s",
        effective_mode,
        list(report.killed_pids) or "[]",
        list(report.survived_pids) or "[]",
        report.gpu_memory_used_mb,
        report.cuda_cache_cleared,
    )
