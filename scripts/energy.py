"""Package-energy metering for one request, and the idle baseline it is measured against.

Why this exists
---------------
A routing decision between the NPU lane and the GPU lane is a decision about energy, and nothing
recorded it. The spend log knows the lane, the tokens and the time; it did not know the joules.

Why it is an integral, not a counter
------------------------------------
This machine has no energy counter to difference. ``rocm-smi --showenergy`` reports the amdgpu
energy counter as unsupported, and the hwmon directory exposes ``power1_average``/``power1_input``
but no ``energy1_input``. So energy is integrated from power: the socket power is sampled while a
request is in flight, and the samples are trapezoid-integrated **over their own timestamps**.

Integrating over real timestamps rather than over an assumed uniform grid matters: a request
shorter than one sampling interval yields only the two boundary readings, and those two readings
are still an estimate of the whole request -- mean power times its actual duration -- instead of
being silently integrated over an interval that never happened.

What the number means, and what it does not
-------------------------------------------
The amdgpu hwmon power is the *socket* figure -- rocm-smi calls it "Socket Graphics Package Power"
-- so it covers the whole APU, not one lane. That makes it comparable *between* lanes, which is
what a routing decision needs, but it does not isolate the NPU. The idle baseline is recorded next
to every measurement so marginal energy (``joules - idle_w * duration_s``) and total energy can be
told apart, because for short requests the baseline dominates.

Resolution: the sensor itself is a driver-averaged reading (an amdgpu averaging window of about a
second), so a request shorter than that window is smeared with what came before it. The sample
count and interval are recorded with every measurement so this is visible in the data rather than
buried here.

The meter never fails a request. If the sensor is absent the record carries ``joules=None`` and a
reason, and the request proceeds.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Sequence, Tuple

logger = logging.getLogger("npuhalo.energy")

# Sampling interval for the in-request integral. 5 Hz resolves a lane change inside a request that
# lasts a second or more, and a sysfs read costs microseconds.
DEFAULT_INTERVAL_S = 0.2
# The idle baseline is re-read at this rate while no request is in flight.
DEFAULT_IDLE_REFRESH_S = 2.0
# An idle reading older than this is not trusted as a baseline; the measurement then falls back to
# the reading taken at request start and says so.
IDLE_MAX_AGE_S = 15.0

# Override for tests, containers with an unusual layout, or a different sensor entirely.
SENSOR_ENV = "NPUHALO_POWER_PATH"

Point = Tuple[float, float]  # (monotonic seconds, watts)


def discover_sensor() -> Optional[str]:
    """Path to the amdgpu power sensor, or None when there is none to read.

    Prefers ``power1_input`` (instantaneous) and falls back to ``power1_average`` (driver-averaged,
    ~1 s window) so the integral works on either. On this machine both files report the same value.
    """
    override = os.environ.get(SENSOR_ENV)
    if override:
        return override if os.path.exists(override) else None
    for hwmon in sorted(glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*")):
        try:
            with open(os.path.join(hwmon, "name")) as handle:
                if handle.read().strip() != "amdgpu":
                    continue
        except OSError:
            continue
        for candidate in ("power1_input", "power1_average"):
            path = os.path.join(hwmon, candidate)
            if os.path.exists(path):
                return path
    return None


def read_watts(path: str) -> Optional[float]:
    """One sensor reading in watts, or None if it cannot be read. hwmon reports microwatts."""
    try:
        with open(path) as handle:
            return int(handle.read().strip()) / 1_000_000.0
    except (OSError, ValueError):
        return None


def integrate_series(points: Sequence[Point]) -> float:
    """Trapezoid integral over timestamped readings, in joules.

    Zero readings integrate to zero. One reading has no width of its own, so it is treated as a
    single default interval wide -- the honest reading of "we saw this power, and nothing else".
    """
    if not points:
        return 0.0
    if len(points) == 1:
        return points[0][1] * DEFAULT_INTERVAL_S
    joules = 0.0
    for (t_before, w_before), (t_after, w_after) in zip(points, points[1:]):
        joules += (w_before + w_after) / 2.0 * max(0.0, t_after - t_before)
    return joules


def integrate(samples: Sequence[float], interval_s: float) -> float:
    """Trapezoid integral of a uniformly spaced power series, in joules.

    Kept for callers that have no timestamps; the meter itself uses :func:`integrate_series`.
    """
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0] * interval_s
    points = [(index * interval_s, watts) for index, watts in enumerate(samples)]
    return integrate_series(points)


@dataclass
class EnergyMeasurement:
    """What one request cost, with the caveats attached rather than implied."""

    joules: Optional[float] = None
    idle_w: Optional[float] = None
    samples: int = 0
    interval_s: float = DEFAULT_INTERVAL_S
    duration_s: float = 0.0
    concurrent: bool = False
    reason: str = ""

    @property
    def marginal_joules(self) -> Optional[float]:
        """Energy above the idle baseline: what the work itself added."""
        if self.joules is None or self.idle_w is None:
            return None
        return max(0.0, self.joules - self.idle_w * self.duration_s)

    @property
    def mean_watts(self) -> Optional[float]:
        """The average power the socket drew across the request."""
        if self.joules is None or self.duration_s <= 0:
            return None
        return self.joules / self.duration_s

    def as_dict(self) -> dict:
        return {
            "energy_j": None if self.joules is None else round(self.joules, 4),
            "energy_marginal_j": None if self.marginal_joules is None else round(self.marginal_joules, 4),
            "energy_idle_w": None if self.idle_w is None else round(self.idle_w, 3),
            "energy_mean_w": None if self.mean_watts is None else round(self.mean_watts, 3),
            "energy_samples": self.samples,
            "energy_interval_s": self.interval_s,
            "energy_duration_s": round(self.duration_s, 4),
            "energy_concurrent": self.concurrent,
            "energy_note": self.reason,
        }


@dataclass
class EnergyMeter:
    """Samples socket power around a request and integrates it over the request's own timestamps.

    The idle baseline is refreshed by :meth:`idle_loop` while nothing is in flight, so a request
    that follows another request is not measured against the previous request's tail.
    """

    sensor_path: Optional[str] = field(default_factory=discover_sensor)
    interval_s: float = DEFAULT_INTERVAL_S
    idle_refresh_s: float = DEFAULT_IDLE_REFRESH_S
    _idle_w: Optional[float] = None
    _idle_at: float = 0.0
    _in_flight: int = 0
    _task: Optional[asyncio.Task] = None
    _stopping: bool = False

    async def start(self) -> None:
        if self.sensor_path is None:
            logger.info("energy: no amdgpu power sensor found; requests will record energy_j=null")
            return
        self._task = asyncio.create_task(self.idle_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def idle_loop(self) -> None:
        """Keep the idle baseline fresh, but only while no request is in flight."""
        while not self._stopping:
            if self._in_flight == 0 and self.sensor_path:
                watts = read_watts(self.sensor_path)
                if watts is not None:
                    self._idle_w = watts
                    self._idle_at = time.monotonic()
            await asyncio.sleep(self.idle_refresh_s)

    async def _sample_loop(self, points: List[Point], stop: asyncio.Event) -> None:
        """Take a reading every interval until the request finishes."""
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
                return
            except asyncio.TimeoutError:
                pass
            watts = read_watts(self.sensor_path)
            if watts is not None:
                points.append((time.monotonic(), watts))

    def _baseline(self, at_start: Optional[float]) -> Tuple[Optional[float], str]:
        """The baseline to subtract, and a note when it is a fallback rather than a measured idle."""
        fresh = (
            self._idle_w is not None
            and (time.monotonic() - self._idle_at) <= IDLE_MAX_AGE_S
        )
        if fresh:
            return self._idle_w, ""
        if at_start is not None:
            return at_start, "idle baseline not fresh; used the reading at request start"
        return None, "no idle baseline available"

    @asynccontextmanager
    async def measure(self) -> AsyncIterator[EnergyMeasurement]:
        """Meter one request. Never raises for sensor problems."""
        measurement = EnergyMeasurement(interval_s=self.interval_s)
        if self.sensor_path is None:
            measurement.reason = "no amdgpu power sensor"
            yield measurement
            return

        self._in_flight += 1
        measurement.concurrent = self._in_flight > 1
        started = time.monotonic()
        at_start = read_watts(self.sensor_path)
        points: List[Point] = [] if at_start is None else [(started, at_start)]
        stop = asyncio.Event()
        sampler = asyncio.create_task(self._sample_loop(points, stop))
        try:
            yield measurement
        finally:
            # The response is finished here: streaming handlers write every chunk before they
            # return, so this is the end of the request, not the end of the headers.
            stop.set()
            sampler.cancel()
            try:
                await sampler
            except asyncio.CancelledError:
                pass
            finished = time.monotonic()
            last = read_watts(self.sensor_path)
            if last is not None:
                points.append((finished, last))
            self._in_flight -= 1

            measurement.duration_s = finished - started
            measurement.samples = len(points)
            if not points:
                measurement.reason = "power sensor unreadable"
                return
            measurement.joules = integrate_series(points)
            idle_w, note = self._baseline(at_start)
            measurement.idle_w = idle_w
            measurement.reason = note
