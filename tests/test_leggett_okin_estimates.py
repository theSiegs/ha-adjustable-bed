"""Estimated positions on Leggett Okin Prodigy CE (CU170) (#601).

The CU170 moves a motor only while its key streams, releases it with four zero
frames and stops 217-218 ms after the last frame, so its four motors qualify
for travel-time estimates. The other Leggett apps declare none.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant.common import MockConfigEntry

from custom_components.adjustable_bed import position_estimate as pe
from custom_components.adjustable_bed.beds.leggett.okin import LeggettOkinController
from custom_components.adjustable_bed.codec import okin_nframe as codec
from custom_components.adjustable_bed.const import (
    BED_TYPE_LEGGETT_OKIN,
    CONF_BED_TYPE,
    CONF_DISABLE_ANGLE_SENSING,
    CONF_LEGGETT_APP_PROFILE,
    CONF_PREFERRED_ADAPTER,
    DOMAIN,
)
from custom_components.adjustable_bed.position_estimate import EstimateState
from custom_components.adjustable_bed.position_estimate_flow import (
    eligible_motors,
    not_offered_motors,
)
from custom_components.adjustable_bed.profiles import okin_nframe as profiles
from tests.test_okin_nframe_concurrent import (
    ADDRESS,
    _until,
    clock,  # noqa: F401 - imported pytest fixture
    make_runtime,  # noqa: F401 - imported pytest fixture
)

FLOW_ADDRESS = "AA:BB:CC:DD:EE:01"
MOTORS = ["head", "feet", "lumbar", "pillow"]
LUMBAR = codec.frame(profiles.LEGGETT_PROFILES["prodigy4"].keys["lumbar_up"]).hex()


def _controller(app_profile: str) -> LeggettOkinController:
    """A controller of one app profile, without a coordinator or a link."""
    controller = object.__new__(LeggettOkinController)
    controller.profile = profiles.LEGGETT_PROFILES[app_profile]  # type: ignore[assignment]
    controller._app_profile = app_profile
    return controller


def test_prodigy_ce_declares_each_motor_on_its_own() -> None:
    controller = _controller("prodigy4")
    assert controller.estimated_position_motor_groups == {
        "head": ("head",),
        "lumbar": ("lumbar",),
        "pillow": ("pillow",),
        "feet": ("feet",),
    }
    assert eligible_motors(controller) == MOTORS
    assert not_offered_motors(controller) == []


@pytest.mark.parametrize("app_profile", ["prodigy2l", "prodigy2", "useries"])
def test_other_leggett_apps_declare_no_estimates(app_profile: str) -> None:
    """Without the CU170's hardware evidence a motor is listed as not offered."""
    controller = _controller(app_profile)
    assert controller.estimated_position_motor_groups == {}
    assert eligible_motors(controller) == []
    assert not_offered_motors(controller) == [spec.key for spec in controller.motor_control_specs]


async def _timed_move(hass: HomeAssistant, motor: str, direction: str, duration_ms: int) -> None:
    await hass.services.async_call(
        DOMAIN,
        "timed_move",
        {"device_id": ["bed"], "motor": motor, "direction": direction, "duration_ms": duration_ms},
        blocking=True,
    )


async def test_every_motor_follows_its_own_direction_time(
    make_runtime,  # noqa: F811 - pytest fixture argument
    clock,  # noqa: F811 - pytest fixture argument
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lumbar and pillow stream together; each uses its own up or down time."""
    monkeypatch.setattr(pe, "_clock", clock.time)
    hass, coordinator, frames = make_runtime("prodigy4")
    entry_id = coordinator.entry.entry_id
    manager = await pe.async_setup_position_estimates(hass, entry_id, ((ADDRESS, None),))
    bed = manager.beds[ADDRESS]
    bed.configure(
        {
            "head": (20.0, 10.0),
            "feet": (20.0, 10.0),
            "lumbar": (10.0, 5.0),
            "pillow": (8.0, 4.0),
        }
    )
    bed.set_reference()
    try:
        lumbar = asyncio.create_task(_timed_move(hass, "lumbar", "up", 1000))
        await _until(frames, LUMBAR)
        await asyncio.wait_for(_timed_move(hass, "pillow", "up", 400), 3)
        await asyncio.wait_for(lumbar, 3)
        assert bed.motors["lumbar"].percent == pytest.approx(10.0)
        assert bed.motors["pillow"].percent == pytest.approx(5.0)
        await asyncio.wait_for(_timed_move(hass, "lumbar", "down", 250), 3)
        # Lowering uses the lumbar's own 5 s: 250 ms is 5%.
        assert bed.motors["lumbar"].percent == pytest.approx(5.0)
        assert bed.tracking(MOTORS)
        assert bed.motors["head"].percent == 0.0
    finally:
        await pe.async_unload_position_estimates(hass, entry_id)


@pytest.fixture
async def entry(
    hass: HomeAssistant,
    mock_coordinator_connected: None,
    enable_custom_integrations: None,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Test bed",
        data={
            "address": FLOW_ADDRESS,
            "name": "Test bed",
            CONF_BED_TYPE: BED_TYPE_LEGGETT_OKIN,
            CONF_LEGGETT_APP_PROFILE: "prodigy4",
            CONF_DISABLE_ANGLE_SENSING: True,
            CONF_PREFERRED_ADAPTER: "auto",
        },
        unique_id=FLOW_ADDRESS,
        entry_id="leggett_estimate_entry",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_typed_power_meter_times_enable_every_motor(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """Times measured elsewhere are typed in the review, to 0.1 s."""
    options = hass.config_entries.options
    result = await options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    assert "estimated_positions" in result["menu_options"]
    result = await options.async_configure(
        result["flow_id"], {"next_step_id": "estimated_positions"}
    )
    assert result["type"] is FlowResultType.FORM
    offered = [str(marker) for marker in result["data_schema"].schema]
    assert offered[:4] == MOTORS
    assert result["description_placeholders"]["not_offered"] == "-"
    result = await options.async_configure(
        result["flow_id"],
        {**dict.fromkeys(MOTORS, True), "measure": False, "acknowledge": True},
    )
    assert result["step_id"] == "estimate_review"
    times = {
        "head_up": 24.3,
        "head_down": 14.9,
        "feet_up": 23.4,
        "feet_down": 19.6,
        "lumbar_up": 15.2,
        "lumbar_down": 11.3,
        "pillow_up": 11.9,
        "pillow_down": 12.4,
    }
    result = await options.async_configure(result["flow_id"], times)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    manager = pe.position_estimates(hass, entry.entry_id)
    assert manager is not None
    bed = manager.beds[FLOW_ADDRESS]
    assert {motor: (e.up_s, e.down_s) for motor, e in bed.motors.items()} == {
        motor: (times[f"{motor}_up"], times[f"{motor}_down"]) for motor in MOTORS
    }
    # Typed times alone are no reference.
    assert all(bed.state(motor) is EstimateState.UNKNOWN for motor in MOTORS)
    registry = er.async_get(hass)
    keys = {
        entity.translation_key
        for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
    }
    assert {f"{motor}_position_estimated" for motor in MOTORS} <= keys
    assert {f"set_{motor}_reference" for motor in MOTORS} <= keys


# Full travel from flat, (up, down) seconds, as a power meter measured on a CU170.
CU170_TIMES = {
    "head": (34.0, 36.0),
    "feet": (20.5, 17.5),
    "lumbar": (13.0, 9.5),
    "pillow": (12.5, 9.0),
}
FLAT = codec.frame(profiles.LEGGETT_PROFILES["prodigy4"].keys["flat"]).hex()


async def _estimated_runtime(build: Any, virtual: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(pe, "_clock", virtual.time)
    hass, coordinator, frames = build("prodigy4")
    manager = await pe.async_setup_position_estimates(
        hass, coordinator.entry.entry_id, ((ADDRESS, None),)
    )
    bed = manager.beds[ADDRESS]
    bed.configure(dict(CU170_TIMES))
    return hass, coordinator, frames, bed


async def _flat(hass: HomeAssistant, coordinator: Any) -> None:
    await coordinator.async_execute_controller_command(
        pe.track_flat_preset(hass, lambda ctrl: ctrl.preset_flat()), cancel_running=True
    )


async def test_a_completed_flat_references_every_motor_it_can_lower(
    make_runtime,  # noqa: F811 - pytest fixture argument
    clock,  # noqa: F811 - pytest fixture argument
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flat holds every motor down for 30 s; a motor that lowers within it is flat.

    The head needs 36 s from the top, plus the margin, so from an unknown
    position the 30 s Flat cannot prove it flat.
    """
    hass, coordinator, _frames, bed = await _estimated_runtime(make_runtime, clock, monkeypatch)
    try:
        assert all(bed.state(motor) is EstimateState.UNKNOWN for motor in MOTORS)
        await asyncio.wait_for(_flat(hass, coordinator), 40)
        assert bed.tracking(["feet", "lumbar", "pillow"])
        assert all(bed.motors[motor].percent == 0.0 for motor in ("feet", "lumbar", "pillow"))
        assert bed.state("head") is EstimateState.UNKNOWN
        assert bed.motors["head"].reason is pe.EstimateReason.HARDWARE_PRESET
    finally:
        await pe.async_unload_position_estimates(hass, coordinator.entry.entry_id)


async def test_a_tracked_motor_only_has_to_lower_from_its_estimate(
    make_runtime,  # noqa: F811 - pytest fixture argument
    clock,  # noqa: F811 - pytest fixture argument
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass, coordinator, _frames, bed = await _estimated_runtime(make_runtime, clock, monkeypatch)
    try:
        bed.set_reference()
        await asyncio.wait_for(_timed_move(hass, "head", "up", 17000), 30)
        assert bed.motors["head"].percent == pytest.approx(50.0)
        await asyncio.wait_for(_flat(hass, coordinator), 40)
        # 50% of 36 s, plus the margin, fits in the Flat.
        assert bed.tracking(MOTORS)
        assert all(bed.motors[motor].percent == 0.0 for motor in MOTORS)
        assert bed.motors["head"].moves_since_reference == 0
    finally:
        await pe.async_unload_position_estimates(hass, coordinator.entry.entry_id)


async def test_a_stopped_flat_proves_nothing(
    make_runtime,  # noqa: F811 - pytest fixture argument
    clock,  # noqa: F811 - pytest fixture argument
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass, coordinator, frames, bed = await _estimated_runtime(make_runtime, clock, monkeypatch)
    try:
        bed.set_reference()
        flat = asyncio.create_task(_flat(hass, coordinator))
        await _until(frames, FLAT)
        await asyncio.sleep(25)
        await coordinator.async_stop_command()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(flat, 5)
        assert all(bed.state(motor) is EstimateState.UNKNOWN for motor in MOTORS)
    finally:
        await pe.async_unload_position_estimates(hass, coordinator.entry.entry_id)


async def test_a_longer_flat_hold_references_the_head_too(
    make_runtime,  # noqa: F811 - pytest fixture argument
    clock,  # noqa: F811 - pytest fixture argument
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hold_control can hold Flat past the head's 36 s from the top."""
    hass, coordinator, _frames, bed = await _estimated_runtime(make_runtime, clock, monkeypatch)
    try:
        await asyncio.wait_for(
            hass.services.async_call(
                DOMAIN,
                "hold_control",
                {"device_id": ["bed"], "control": "flat", "duration": 45},
                blocking=True,
            ),
            60,
        )
        assert bed.tracking(MOTORS)
        assert all(bed.motors[motor].percent == 0.0 for motor in MOTORS)
    finally:
        await pe.async_unload_position_estimates(hass, coordinator.entry.entry_id)


async def test_the_flat_button_references_the_bed(
    hass: HomeAssistant, entry: MockConfigEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preset_flat button goes through the Flat reference, not a plain invalidation."""
    manager = pe.position_estimates(hass, entry.entry_id)
    assert manager is not None
    bed = manager.beds[FLOW_ADDRESS]
    bed.configure(dict(CU170_TIMES))
    flat = AsyncMock()
    monkeypatch.setattr(LeggettOkinController, "preset_flat", flat)
    ticks = itertools.count(step=31.0)
    monkeypatch.setattr(pe, "_clock", lambda: next(ticks))
    registry = er.async_get(hass)
    entity_id = next(
        entity.entity_id
        for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        if entity.translation_key == "preset_flat"
    )
    await hass.services.async_call("button", "press", {"entity_id": entity_id}, blocking=True)
    flat.assert_awaited_once()
    assert bed.tracking(["feet", "lumbar", "pillow"])
    assert bed.state("head") is EstimateState.UNKNOWN


class _Parent:
    """Stands in for the loaded bed: timed moves end at once, STOP bumps the epoch."""

    def __init__(self, bed: pe.BedEstimate) -> None:
        self._bed = bed
        self._command_scheduler = SimpleNamespace(stop_epoch=0)
        self.moves: list[tuple[str, str, int]] = []
        self.stop_after_s: float | None = None

    async def timed_move(
        self, _targets: object, motor: str, direction: str, duration_ms: int
    ) -> None:
        self.moves.append((motor, direction, duration_ms))
        token = self._bed.begin_move((motor,))
        seconds = duration_ms / 1000
        if self.stop_after_s is not None:
            seconds = self.stop_after_s
            self._command_scheduler.stop_epoch += 1
        self._bed.end_move(token, direction, seconds)


async def _end_stop_bed(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> tuple[pe.PositionEstimates, pe.BedEstimate, _Parent]:
    manager = await pe.async_setup_position_estimates(hass, "entry", ((FLOW_ADDRESS, None),))
    bed = manager.beds[FLOW_ADDRESS]
    bed.configure(dict(CU170_TIMES))
    bed.set_reference()
    parent = _Parent(bed)
    hass.data[DOMAIN]["entry"] = parent
    monkeypatch.setattr(
        "custom_components.adjustable_bed.services._async_timed_move", parent.timed_move
    )
    return manager, bed, parent


async def test_an_end_target_overruns_to_the_end_stop(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0% and 100% run the full travel plus the margin, whatever the estimate says."""
    manager, bed, parent = await _end_stop_bed(hass, monkeypatch)
    bed.motors["feet"].percent = 40.0
    assert await pe.async_move_bed(hass, manager, bed, {"feet": 100.0})
    bed.motors["head"].percent = 60.0
    bed.motors["head"].moves_since_reference = 3
    assert await pe.async_move_bed(hass, manager, bed, {"head": 0.0})
    # A target between the ends keeps the exact plan.
    assert await pe.async_move_bed(hass, manager, bed, {"lumbar": 50.0})
    assert parent.moves == [
        ("feet", "up", 25600),  # 20.5 s x 1.2 + 1 s
        ("head", "down", 44200),  # 36 s x 1.2 + 1 s
        ("lumbar", "up", 6500),
    ]
    assert bed.motors["feet"].percent == 100.0
    assert bed.motors["head"].percent == 0.0
    # Reaching the bottom end stop re-references the motor.
    assert bed.motors["head"].moves_since_reference == 0
    assert bed.motors["lumbar"].percent == pytest.approx(50.0)
    assert bed.tracking(MOTORS)


async def test_a_stopped_overrun_does_not_re_reference(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, bed, parent = await _end_stop_bed(hass, monkeypatch)
    bed.motors["pillow"].percent = 10.0
    parent.stop_after_s = 5.0
    await pe.async_move_bed(hass, manager, bed, {"pillow": 0.0})
    # The estimate reaches 0%, but only a full overrun proves the end stop.
    assert bed.motors["pillow"].percent == 0.0
    assert bed.motors["pillow"].moves_since_reference == 1


async def test_connecting_and_gaps_between_moves_are_not_travel(
    make_runtime,  # noqa: F811 - pytest fixture argument
    clock,  # noqa: F811 - pytest fixture argument
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the time the motor is commanded counts, from its first frame to its release.

    A slow (re)connect before each move and the pause between two chained
    timed moves add nothing to the estimate.
    """
    hass, coordinator, frames, bed = await _estimated_runtime(make_runtime, clock, monkeypatch)

    async def slow_connect(*_args: object, **_kwargs: object) -> bool:
        await asyncio.sleep(5)
        return True

    monkeypatch.setattr(coordinator, "async_ensure_connected", slow_connect)
    try:
        bed.set_reference()
        started = clock.time()
        await asyncio.wait_for(_timed_move(hass, "lumbar", "up", 1300), 30)
        await asyncio.sleep(3.5)
        await asyncio.wait_for(_timed_move(hass, "lumbar", "up", 1300), 30)
        # Two connects and the pause took far longer than the 2.6 s of travel.
        assert clock.time() - started > 10
        assert bed.motors["lumbar"].percent == pytest.approx(20.0)  # 2.6 s of 13 s
        assert bed.motors["lumbar"].moves_since_reference == 2
    finally:
        await pe.async_unload_position_estimates(hass, coordinator.entry.entry_id)
