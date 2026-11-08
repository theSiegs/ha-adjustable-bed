"""Estimated positions on Leggett Okin Prodigy CE (CU170) (#601).

The CU170 moves a motor only while its key streams, releases it with four zero
frames and stops 217-218 ms after the last frame, so its four motors qualify
for travel-time estimates. The other Leggett apps declare none.
"""

from __future__ import annotations

import asyncio

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
