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
"""Tests for lelab.sim — the fake SO-101 devices used by `lelab --sim`."""

from __future__ import annotations

import pytest


@pytest.fixture
def sim_on(monkeypatch: pytest.MonkeyPatch):
    from lelab import sim

    monkeypatch.setenv(sim.SIM_ENV_VAR, "1")
    return sim


def test_sim_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import sim

    monkeypatch.delenv(sim.SIM_ENV_VAR, raising=False)
    assert sim.sim_enabled() is False


def test_sim_enabled_accepts_common_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import sim

    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(sim.SIM_ENV_VAR, value)
        assert sim.sim_enabled() is True
    monkeypatch.setenv(sim.SIM_ENV_VAR, "0")
    assert sim.sim_enabled() is False


def test_available_ports_are_sim_ports_when_enabled(sim_on) -> None:
    from lelab.utils.config import find_available_ports

    assert find_available_ports() == sim_on.SIM_PORTS


def test_detect_port_after_disconnect_returns_a_sim_port(sim_on, monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab.utils import config

    monkeypatch.setattr(config.time, "sleep", lambda _s: None)
    assert config.detect_port_after_disconnect(sim_on.SIM_PORTS) in sim_on.SIM_PORTS


def test_bus_positions_move_over_time(sim_on) -> None:
    bus = sim_on.SimBus(sim_on.SIM_FOLLOWER_PORT)
    bus.connect()
    first = bus.sync_read("Present_Position", normalize=False)
    bus._t0 -= 1.0  # advance the sweep a second without sleeping
    second = bus.sync_read("Present_Position", normalize=False)

    assert set(first) == set(sim_on.SO101_MOTORS)
    assert first != second
    # Raw readings must stay inside the encoder range calibration validates.
    assert all(0 < pos < sim_on.ENCODER_RESOLUTION for pos in first.values())


def test_bus_sweep_covers_a_calibratable_range(sim_on) -> None:
    """One full period must move every joint further than calibration's
    "did it actually move" floor, and never jump far enough per 20 Hz sample to
    look like an encoder wrap-around."""
    pytest.importorskip("lerobot")  # lelab.calibrate imports it
    from lelab.calibrate import _MAX_POSITION_JUMP, _MIN_CALIBRATION_RANGE

    bus = sim_on.SimBus(sim_on.SIM_FOLLOWER_PORT)
    bus.connect()
    samples = []
    for step in range(200):  # 10 s at 20 Hz, longer than the slowest period
        bus._t0 = -step * 0.05
        samples.append(bus.sync_read("Present_Position", normalize=False))

    for motor in sim_on.SO101_MOTORS:
        values = [s[motor] for s in samples]
        assert max(values) - min(values) > _MIN_CALIBRATION_RANGE
        jumps = [abs(b - a) for a, b in zip(values, values[1:], strict=True)]
        assert max(jumps) < _MAX_POSITION_JUMP


def test_half_turn_homings_center_the_reading(sim_on) -> None:
    bus = sim_on.SimBus(sim_on.SIM_FOLLOWER_PORT)
    positions = {"shoulder_pan": 2500}
    assert bus._get_half_turn_homings(positions) == {"shoulder_pan": 2500 - sim_on.ENCODER_CENTER}


@pytest.fixture
def sim_devices(sim_on):
    """The sim device classes, skipped when lerobot isn't installed.

    They subclass the real SO-101 devices, so building them needs lerobot.
    """
    pytest.importorskip("lerobot")
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    return sim_on, SO101FollowerConfig, SO101LeaderConfig


def test_devices_subclass_the_real_lerobot_classes(sim_devices) -> None:
    """`record_loop` branches on `isinstance(teleop, Teleoperator)` and records
    nothing at all when that check fails, so the inheritance is load-bearing."""
    sim, follower_config, leader_config = sim_devices
    from lerobot.robots import Robot
    from lerobot.teleoperators import Teleoperator

    robot = sim.make_follower(follower_config(port=sim.SIM_FOLLOWER_PORT, id="sim_follower"))
    leader = sim.make_leader(leader_config(port=sim.SIM_LEADER_PORT, id="sim_leader"))

    assert isinstance(robot, Robot)
    assert isinstance(leader, Teleoperator)


def test_follower_observation_matches_its_features(sim_devices, tmp_lerobot_home) -> None:
    sim, follower_config, _ = sim_devices
    from lerobot.cameras.opencv import OpenCVCameraConfig

    robot = sim.make_follower(
        follower_config(
            port=sim.SIM_FOLLOWER_PORT,
            id="sim_follower",
            cameras={"front": OpenCVCameraConfig(index_or_path=0, width=320, height=240, fps=30)},
        )
    )
    robot.connect(calibrate=False)

    assert robot.is_connected
    observation = robot.get_observation()
    assert set(observation) == set(robot.observation_features)
    assert observation["front"].shape == (240, 320, 3)
    assert robot.observation_features["front"] == (240, 320, 3)

    robot.disconnect()
    assert not robot.is_connected


def test_follower_is_connected_tracks_bus_and_cameras(sim_devices) -> None:
    """Recording brings the bus and cameras up itself instead of calling
    connect(), and `record_loop` gates on this flag."""
    sim, follower_config, _ = sim_devices
    from lerobot.cameras.opencv import OpenCVCameraConfig

    robot = sim.make_follower(
        follower_config(
            port=sim.SIM_FOLLOWER_PORT,
            id="sim_follower",
            cameras={"front": OpenCVCameraConfig(index_or_path=0, width=320, height=240, fps=30)},
        )
    )

    robot.bus.connect()
    assert not robot.is_connected  # camera still down
    robot.cameras["front"].connect()
    assert robot.is_connected


def test_leader_action_matches_follower_action_features(sim_devices) -> None:
    sim, follower_config, leader_config = sim_devices

    leader = sim.make_leader(leader_config(port=sim.SIM_LEADER_PORT, id="sim_leader"))
    follower = sim.make_follower(follower_config(port=sim.SIM_FOLLOWER_PORT, id="sim_follower"))
    leader.bus.connect()
    follower.bus.connect()

    action = leader.get_action()
    assert set(action) == set(follower.action_features)
    assert set(follower.send_action(action)) == set(action)


def test_leader_and_follower_are_out_of_phase(sim_devices) -> None:
    sim, follower_config, leader_config = sim_devices

    leader = sim.make_leader(leader_config(port=sim.SIM_LEADER_PORT, id="sim_leader"))
    follower = sim.make_follower(follower_config(port=sim.SIM_FOLLOWER_PORT, id="sim_follower"))
    leader.bus.connect()
    follower.bus.connect()
    leader.bus._t0 = follower.bus._t0

    assert leader.get_action() != follower.get_observation()


def test_make_device_from_config_picks_the_side(sim_devices) -> None:
    sim, follower_config, leader_config = sim_devices

    leader = sim.make_device_from_config(leader_config(port=sim.SIM_LEADER_PORT, id="sim_leader"))
    follower = sim.make_device_from_config(follower_config(port=sim.SIM_FOLLOWER_PORT, id="sim_follower"))

    assert leader.name == "so_leader"
    assert follower.name == "so_follower"


def test_sim_bus_never_asks_for_interactive_calibration(sim_devices) -> None:
    """LeRobot's connect() falls back to an input()-driven calibrate() when the
    bus reports itself uncalibrated — which would hang the server."""
    sim, _, leader_config = sim_devices

    leader = sim.make_leader(leader_config(port=sim.SIM_LEADER_PORT, id="sim_leader"))
    assert leader.bus.is_calibrated is True


def test_camera_frames_change_between_reads(sim_on) -> None:
    camera = sim_on.SimCamera("front", 64, 48, 30)
    camera.connect()
    first = camera.read_latest()
    camera._t0 -= 0.5
    second = camera.read_latest()

    assert first.shape == (48, 64, 3)
    assert first.dtype.name == "uint8"
    assert not (first == second).all()


def test_sim_cameras_listing_is_non_empty(sim_on) -> None:
    cameras = sim_on.sim_cameras()
    assert cameras and all("index" in cam and "name" in cam for cam in cameras)
