# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simulated SO-101 hardware, for developing the interface without arms.

Enabled with ``lelab --sim`` (or ``LELAB_SIM=1``). When on, the feature modules
build these fakes instead of the real ``SO101Leader`` / ``SO101Follower``:
everything downstream — calibration, teleoperation, the joint-data broadcast,
the record phase machine, the dataset written to disk — runs through its normal
code path, it just reads a sine-driven arm and synthetic camera frames instead
of a serial bus.

The fakes mirror the LeRobot device surface leLab and ``record_loop`` actually
touch (``bus``, ``cameras``, ``calibration``, ``get_observation``,
``get_action``, ``send_action``, ``*_features``), not the whole class.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SIM_ENV_VAR = "LELAB_SIM"

# Fake serial ports. They look like device nodes so the frontend's port pickers
# render them normally, and the "lelab-sim" infix makes it obvious in the UI (and
# in a recorded dataset's metadata) that no real arm was involved.
SIM_LEADER_PORT = "/dev/tty.lelab-sim-leader"
SIM_FOLLOWER_PORT = "/dev/tty.lelab-sim-follower"
SIM_PORTS = [SIM_FOLLOWER_PORT, SIM_LEADER_PORT]

# SO-101 motor order, ids and model, as the real bus reports them.
SO101_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SO101_MODEL = "sts3215"

# Feetech sts3215 encoders are 12-bit; calibration reads raw steps in 0..4095
# and treats mid-scale as the half-turn home.
ENCODER_RESOLUTION = 4096
ENCODER_CENTER = ENCODER_RESOLUTION // 2

# Sine sweep per joint: (amplitude in encoder steps, period in seconds). The
# amplitudes clear calibration's 100-step "did the joint actually move" check
# within one period, and stay far short of the 2000-step jump that reads as an
# encoder wrap-around at the 20 Hz sampling calibration uses.
_MOTION = {
    "shoulder_pan": (900, 7.0),
    "shoulder_lift": (700, 5.0),
    "elbow_flex": (800, 6.0),
    "wrist_flex": (600, 4.0),
    "wrist_roll": (850, 9.0),
    "gripper": (500, 3.0),
}

_DEFAULT_CAMERA_SIZE = (640, 480)  # width, height
_DEFAULT_CAMERA_FPS = 30


def sim_enabled() -> bool:
    """True when this process was started in simulation mode."""
    return os.environ.get(SIM_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class _SimMotor:
    """Stands in for ``lerobot.motors.Motor`` (only ``id``/``model`` are read)."""

    id: int
    model: str = SO101_MODEL


def _raw_position(motor: str, t: float, phase: float) -> int:
    """Raw encoder reading for `motor` at time `t`, on a per-arm phase offset."""
    amplitude, period = _MOTION[motor]
    return int(ENCODER_CENTER + amplitude * math.sin(2 * math.pi * t / period + phase))


def _degrees(motor: str, t: float, phase: float) -> float:
    """Position in the units ``get_observation`` reports.

    LeRobot drives the SO-101 with ``use_degrees=True``: every joint is degrees
    from the calibration center, except the gripper, which is a 0..100 opening
    percentage. Derived from the same sweep as the raw reading so the 3D viewer
    and a recorded episode show the same motion.
    """
    amplitude, period = _MOTION[motor]
    wave = math.sin(2 * math.pi * t / period + phase)
    if motor == "gripper":
        return round(50.0 + 50.0 * wave, 3)
    return round(amplitude / ENCODER_RESOLUTION * 360.0 * wave, 3)


class SimBus:
    """Fake Feetech bus: sine-driven positions and a register write log."""

    def __init__(self, port: str, phase: float = 0.0) -> None:
        self.port = port
        self.motors = {name: _SimMotor(id=i + 1) for i, name in enumerate(SO101_MOTORS)}
        self.calibration: dict[str, Any] = {}
        self.torque_enabled = False
        # Kept so a test (or a curious developer) can see what the calibration
        # flow wrote; nothing in leLab reads it back.
        self.written_registers: list[tuple[str, str, Any]] = []
        self.port_handler = None
        self._phase = phase
        self._connected = False
        self._t0 = time.monotonic()

    # --- lifecycle ---------------------------------------------------------
    def connect(self, *args: Any, **kwargs: Any) -> None:
        self._connected = True
        self._t0 = time.monotonic()
        logger.info("[sim] bus connected on %s", self.port)

    def disconnect(self, *args: Any, **kwargs: Any) -> None:
        self._connected = False
        logger.info("[sim] bus disconnected on %s", self.port)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # --- registers ---------------------------------------------------------
    def _elapsed(self) -> float:
        return time.monotonic() - self._t0

    def sync_read(self, data_name: str, motors: Any = None, *, normalize: bool = True, **kwargs: Any):
        names = list(self.motors) if motors is None else list(motors)
        if data_name != "Present_Position":
            return dict.fromkeys(names, 0)
        t = self._elapsed()
        if normalize:
            return {m: _degrees(m, t, self._phase) for m in names}
        return {m: _raw_position(m, t, self._phase) for m in names}

    def read(self, data_name: str, motor: str, *, normalize: bool = True, **kwargs: Any):
        return self.sync_read(data_name, [motor], normalize=normalize)[motor]

    def write(self, data_name: str, motor: str, value: Any, **kwargs: Any) -> None:
        self.written_registers.append((data_name, motor, value))

    def sync_write(self, data_name: str, values: dict[str, Any], **kwargs: Any) -> None:
        for motor, value in values.items():
            self.write(data_name, motor, value)

    def enable_torque(self, *args: Any, **kwargs: Any) -> None:
        self.torque_enabled = True

    def disable_torque(self, *args: Any, **kwargs: Any) -> None:
        self.torque_enabled = False

    def configure_motors(self, *args: Any, **kwargs: Any) -> None:
        pass

    @contextmanager
    def torque_disabled(self):
        """Mirrors the real bus context manager `configure()` runs inside."""
        self.disable_torque()
        try:
            yield
        finally:
            self.enable_torque()

    # --- calibration -------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        # Always "calibrated", so `connect()` never drops into LeRobot's
        # interactive calibrate() — which would block on stdin under uvicorn.
        return True

    def reset_calibration(self, *args: Any, **kwargs: Any) -> None:
        self.calibration = {}

    def set_half_turn_homings(self, *args: Any, **kwargs: Any) -> dict[str, int]:
        return self._get_half_turn_homings(self.sync_read("Present_Position", normalize=False))

    def write_calibration(self, calibration: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.calibration = dict(calibration or {})

    def _get_half_turn_homings(self, positions: dict[str, float]) -> dict[str, int]:
        """Same contract as the real bus: offset that puts each joint's current
        reading at mid-scale."""
        return {motor: int(pos) - ENCODER_CENTER for motor, pos in positions.items()}


class SimCamera:
    """Fake camera producing a moving synthetic RGB frame.

    Frames are generated on demand from the clock, so the MJPEG preview, the
    record loop and the dataset all see a scene that actually moves: a scrolling
    vertical band over a per-camera color wash, plus a marker block that tracks
    the same sine the arm follows.
    """

    def __init__(self, name: str, width: int, height: int, fps: int) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self._connected = False
        self._t0 = time.monotonic()
        # Stable per-camera hue so two configured cameras are told apart at a glance.
        self._hue = (abs(hash(name)) % 6) / 6.0

    def connect(self, *args: Any, **kwargs: Any) -> None:
        self._connected = True
        self._t0 = time.monotonic()
        logger.info("[sim] camera %s connected (%dx%d@%dfps)", self.name, self.width, self.height, self.fps)

    def disconnect(self, *args: Any, **kwargs: Any) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _frame(self):
        import numpy as np

        t = time.monotonic() - self._t0
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Background wash: a vertical gradient tinted by this camera's hue.
        gradient = np.linspace(30, 140, h, dtype=np.uint8)[:, None]
        for c in range(3):
            tint = 0.4 + 0.6 * ((self._hue + c / 3.0) % 1.0)
            frame[:, :, c] = (gradient * tint).astype(np.uint8)

        # Scrolling band, so a dropped/frozen stream is obvious.
        band_w = max(8, w // 16)
        x = int((t * w / 3.0) % (w + band_w)) - band_w
        x0, x1 = max(0, x), max(0, min(w, x + band_w))
        if x1 > x0:
            frame[:, x0:x1, :] = 235

        # Marker block riding the same sweep as the arm, as a crude "scene".
        cx = int(w / 2 + (w / 3) * math.sin(2 * math.pi * t / 6.0))
        cy = int(h / 2 + (h / 4) * math.sin(2 * math.pi * t / 4.0))
        size = max(10, h // 10)
        y0, y1 = max(0, cy - size), min(h, cy + size)
        bx0, bx1 = max(0, cx - size), min(w, cx + size)
        frame[y0:y1, bx0:bx1] = (245, 120, 40)
        return frame

    def read(self, *args: Any, **kwargs: Any):
        return self._frame()

    def async_read(self, *args: Any, **kwargs: Any):
        return self._frame()

    def read_latest(self, *args: Any, **kwargs: Any):
        return self._frame()


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------
#
# These subclass the real LeRobot devices and swap only the bus and cameras.
# Subclassing (rather than duck-typing) is required, not cosmetic: `record_loop`
# branches on `isinstance(teleop, Teleoperator)` and silently records nothing
# when the check fails. It also means every behaviour built on top of the bus —
# `get_observation`, `send_action`, `configure`, the calibration file handling —
# is LeRobot's own code, so sim exercises the real implementations.


@functools.lru_cache(maxsize=1)
def _device_classes() -> tuple[type, type]:
    """Build the sim device classes (imported lazily: lerobot is heavy)."""
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.teleoperators.so_leader import SO101Leader

    class SimFollower(SO101Follower):
        """SO-101 follower whose motors and cameras are simulated."""

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.bus = SimBus(getattr(config, "port", None) or SIM_FOLLOWER_PORT)
            self.cameras = _cameras_from_config(config)

    class SimLeader(SO101Leader):
        """SO-101 leader whose motors are simulated.

        Runs a quarter-period out of phase with the follower so teleoperation
        shows the follower tracking a leader that is actually moving.
        """

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.bus = SimBus(getattr(config, "port", None) or SIM_LEADER_PORT, phase=math.pi / 4)

    return SimFollower, SimLeader


def _cameras_from_config(config: Any) -> dict[str, SimCamera]:
    """Build a SimCamera per camera the request configured.

    Reads whatever width/height/fps the real camera config carried so the
    dataset's video features match what the user picked in the UI.
    """
    cameras: dict[str, SimCamera] = {}
    for name, cam_cfg in (getattr(config, "cameras", None) or {}).items():
        width = getattr(cam_cfg, "width", None) or _DEFAULT_CAMERA_SIZE[0]
        height = getattr(cam_cfg, "height", None) or _DEFAULT_CAMERA_SIZE[1]
        fps = getattr(cam_cfg, "fps", None) or _DEFAULT_CAMERA_FPS
        cameras[name] = SimCamera(name, int(width), int(height), int(fps))
    return cameras


# ---------------------------------------------------------------------------
# Factories used by the feature modules
# ---------------------------------------------------------------------------


def make_follower(config: Any):
    return _device_classes()[0](config)


def make_leader(config: Any):
    return _device_classes()[1](config)


def make_device_from_config(config: Any):
    """Pick leader/follower from the LeRobot config object's own type name."""
    return make_leader(config) if "Leader" in type(config).__name__ else make_follower(config)


def sim_cameras() -> list[dict[str, Any]]:
    """Camera list for ``/available-cameras`` in sim mode."""
    return [
        {"index": 0, "name": "Sim Camera (front)", "available": True},
        {"index": 1, "name": "Sim Camera (wrist)", "available": True},
    ]
