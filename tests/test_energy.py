"""Tests for the package-energy meter: the arithmetic, and every degradation path.

The meter must never turn a sensor problem into a failed request, so most of these tests are about
what happens when the sensor is absent, unreadable, or the request is concurrent with another.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from energy import (  # noqa: E402
    DEFAULT_INTERVAL_S,
    EnergyMeter,
    discover_sensor,
    integrate,
    integrate_series,
    read_watts,
)


# ---- the integral ---------------------------------------------------------

def test_integrate_constant_power():
    # three readings, two intervals of 0.5 s, at 100 W -> 1 s of 100 W
    assert integrate([100.0, 100.0, 100.0], 0.5) == 100.0


def test_integrate_trapezoid_on_a_ramp():
    # 0 -> 100 W over one interval, then flat: 50 + 100 = 150 J
    assert integrate([0.0, 100.0, 100.0], 1.0) == 150.0


def test_integrate_single_reading_counts_as_one_interval():
    # One reading has no width to integrate over; it is at most one interval wide.
    assert integrate([50.0], 0.2) == 10.0


def test_integrate_nothing_is_zero():
    assert integrate([], 0.2) == 0.0


# ---- the timestamped integral (the one the meter uses) --------------------

def test_integrate_series_uses_real_timestamps():
    # Two boundary readings 7.5 s apart at 100 W: the width is the request, not the interval.
    assert integrate_series([(0.0, 100.0), (7.5, 100.0)]) == 750.0


def test_integrate_series_on_a_ramp():
    assert integrate_series([(0.0, 0.0), (1.0, 100.0), (2.0, 100.0)]) == 150.0


def test_integrate_series_single_reading_is_one_interval_wide():
    assert integrate_series([(0.0, 50.0)]) == 50.0 * DEFAULT_INTERVAL_S


def test_integrate_series_ignores_a_backwards_clock():
    assert integrate_series([(1.0, 100.0), (0.5, 100.0)]) == 0.0


# ---- reading the sensor ---------------------------------------------------

def test_read_watts_converts_microwatts(tmp_path):
    sensor = tmp_path / "power1_input"
    sensor.write_text("18092000\n")
    assert read_watts(str(sensor)) == 18.092


def test_read_watts_on_a_missing_file_is_none(tmp_path):
    assert read_watts(str(tmp_path / "absent")) is None


def test_read_watts_on_garbage_is_none(tmp_path):
    sensor = tmp_path / "power1_input"
    sensor.write_text("not a number\n")
    assert read_watts(str(sensor)) is None


def test_discover_sensor_honours_the_override(tmp_path, monkeypatch):
    sensor = tmp_path / "sensor"
    sensor.write_text("1\n")
    monkeypatch.setenv("NPUHALO_POWER_PATH", str(sensor))
    assert discover_sensor() == str(sensor)

    monkeypatch.setenv("NPUHALO_POWER_PATH", str(tmp_path / "gone"))
    assert discover_sensor() is None


# ---- metering a request ---------------------------------------------------

def test_measure_without_a_sensor_degrades_instead_of_raising():
    async def run():
        meter = EnergyMeter(sensor_path=None)
        async with meter.measure() as measurement:
            pass
        return measurement

    measurement = asyncio.run(run())
    assert measurement.joules is None
    assert measurement.samples == 0
    assert "no amdgpu power sensor" in measurement.reason
    assert measurement.as_dict()["energy_j"] is None


def test_measure_integrates_and_reports_the_sample_count(tmp_path):
    sensor = tmp_path / "power1_input"
    sensor.write_text("100000000\n")  # 100 W

    async def run():
        meter = EnergyMeter(sensor_path=str(sensor), interval_s=0.05)
        async with meter.measure() as measurement:
            await asyncio.sleep(0.02)
        return measurement

    measurement = asyncio.run(run())
    # A reading at request start and one at the end: two readings, the width is the real duration.
    assert measurement.samples == 2
    assert measurement.joules == 100.0 * measurement.duration_s
    assert measurement.duration_s > 0


def test_measure_samples_during_a_long_request(tmp_path):
    """Regression: the meter must take readings *during* the request, not only at its edges.

    The first live run reported 5.3 J for 7.5 s of GPU work (0.7 W) because only the two boundary
    readings were taken and they were integrated over one sampling interval.
    """
    sensor = tmp_path / "power1_input"
    sensor.write_text("100000000\n")  # 100 W

    async def run():
        meter = EnergyMeter(sensor_path=str(sensor), interval_s=0.01)
        async with meter.measure() as measurement:
            await asyncio.sleep(0.05)
        return measurement

    measurement = asyncio.run(run())
    assert measurement.samples >= 4, f"expected periodic samples, got {measurement.samples}"
    assert measurement.joules == 100.0 * measurement.duration_s
    assert measurement.mean_watts == 100.0


def test_overlapping_requests_are_flagged_concurrent(tmp_path):
    sensor = tmp_path / "power1_input"
    sensor.write_text("100000000\n")

    async def run():
        meter = EnergyMeter(sensor_path=str(sensor), interval_s=0.05)
        async with meter.measure() as outer:
            async with meter.measure() as inner:
                await asyncio.sleep(0.01)
        return outer, inner

    outer, inner = asyncio.run(run())
    assert inner.concurrent is True, "the second request must be marked concurrent"
    assert outer.concurrent is False, "the first request was alone when it started"


def test_marginal_energy_subtracts_the_baseline(tmp_path):
    sensor = tmp_path / "power1_input"
    sensor.write_text("100000000\n")  # 100 W while working

    async def run():
        meter = EnergyMeter(sensor_path=str(sensor), interval_s=0.05)
        # A fresh idle baseline of 20 W, as if the machine had been quiet.
        meter._idle_w = 20.0
        import time
        meter._idle_at = time.monotonic()
        async with meter.measure() as measurement:
            await asyncio.sleep(0.01)
        return measurement

    measurement = asyncio.run(run())
    assert measurement.idle_w == 20.0
    # 100 W for the request's real duration, of which 20 W was the baseline.
    assert measurement.joules == 100.0 * measurement.duration_s
    assert measurement.marginal_joules == 80.0 * measurement.duration_s
    assert measurement.as_dict()["energy_marginal_j"] is not None


def test_stale_baseline_falls_back_and_says_so(tmp_path):
    sensor = tmp_path / "power1_input"
    sensor.write_text("100000000\n")

    async def run():
        meter = EnergyMeter(sensor_path=str(sensor), interval_s=0.05)
        # An idle reading from long ago must not be trusted.
        meter._idle_w = 20.0
        meter._idle_at = 0.0
        async with meter.measure() as measurement:
            await asyncio.sleep(0.01)
        return measurement

    measurement = asyncio.run(run())
    assert "not fresh" in measurement.reason
    # Falls back to the reading at request start, which is what the sensor said.
    assert measurement.idle_w == 100.0


def test_default_interval_is_the_documented_one():
    assert DEFAULT_INTERVAL_S == 0.2
