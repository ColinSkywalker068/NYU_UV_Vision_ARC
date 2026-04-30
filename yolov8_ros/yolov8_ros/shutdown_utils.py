"""Shared shutdown / signal handling for YOLOv8 ROS nodes.

Why this exists:

* rclpy's internal signal handling is unreliable inside containers and under
  a busy executor, so each node installs its own SIGINT / SIGTERM handlers.
* We also trap SIGHUP (terminal closes) and SIGTSTP (Ctrl+Z) and route them
  into a proper ``destroy_node`` path. SIGTSTP normally suspends the
  process, which on Jetson leaves GPU memory, serial ports, and USB camera
  handles pinned — exactly what we're trying to avoid. We turn the first
  Ctrl+Z into a graceful shutdown; a second Ctrl+Z within 1s falls through
  to the default (suspend) in case a human really wants it.
* A tiny ``atexit`` hook makes the shutdown idempotent: if Python bails via
  an unhandled exception or ``os._exit``, the GPU/serial is still released.
"""

from __future__ import annotations

import atexit
import signal
import sys
import threading
import time
from typing import Callable, Optional

ShutdownCallback = Callable[[], None]

_CTRLZ_DOUBLE_TAP_WINDOW_S: float = 1.0


class _ShutdownCoordinator:
    """Tracks whether shutdown already happened so handlers stay idempotent."""

    def __init__(self, on_shutdown: ShutdownCallback, logger_name: str) -> None:
        self._on_shutdown: ShutdownCallback = on_shutdown
        self._logger_name: str = logger_name
        self._lock: threading.Lock = threading.Lock()
        self._done: bool = False
        self._last_tstp_at: float = 0.0

    def runShutdown(self, reason: str) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self._printReason(reason)
        try:
            self._on_shutdown()
        except Exception as err:
            sys.stderr.write(f"[{self._logger_name}] shutdown hook error: {err}\n")

    def handleSignal(self, signum: int, _frame) -> None:
        sig_name = _safeSignalName(signum)
        if signum == signal.SIGTSTP:
            now = time.monotonic()
            if now - self._last_tstp_at < _CTRLZ_DOUBLE_TAP_WINDOW_S:
                signal.signal(signal.SIGTSTP, signal.SIG_DFL)
                self._printReason(
                    f"{sig_name} pressed twice — falling through to default suspend"
                )
                self.runShutdown(reason=f"{sig_name} (double-tap)")
                signal.raise_signal(signal.SIGTSTP)
                return
            self._last_tstp_at = now
            self._printReason(
                f"{sig_name} received — shutting down cleanly instead of suspending "
                f"(press Ctrl+Z again within 1s to actually suspend)"
            )
            self.runShutdown(reason=sig_name)
            raise SystemExit(0)
        self.runShutdown(reason=sig_name)
        raise SystemExit(0)

    def _printReason(self, reason: str) -> None:
        sys.stderr.write(f"\n[{self._logger_name}] {reason}\n")


def _safeSignalName(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal({signum})"


def installShutdownHandlers(
    on_shutdown: ShutdownCallback,
    logger_name: str = "node",
) -> None:
    """Install SIGINT / SIGTERM / SIGHUP / SIGTSTP handlers.

    ``on_shutdown`` must be idempotent; it is called at most once by signals
    or ``atexit``, whichever fires first.
    """
    coordinator = _ShutdownCoordinator(on_shutdown=on_shutdown, logger_name=logger_name)
    atexit.register(lambda: coordinator.runShutdown(reason="atexit"))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP):
        try:
            signal.signal(sig, coordinator.handleSignal)
        except (ValueError, OSError):
            # Not the main thread, or signal unsupported on this platform.
            continue


def shutdownRcl(
    node: object | None = None,
    executor: object | None = None,
) -> None:
    """Destroy a node + executor + rclpy context, swallowing per-step errors."""
    if node is not None:
        try:
            node.destroy_node()
        except Exception:
            pass
    if executor is not None:
        try:
            executor.shutdown()
        except Exception:
            pass
    try:
        import rclpy

        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def releaseCudaMemory() -> bool:
    """Best-effort torch.cuda release. Safe to call when torch isn't installed."""
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
    except Exception:
        return False
