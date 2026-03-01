"""
core/machines.py
SimPy-based Machine model with:
  - Discrete-event processing (cycle time, fault, repair)
  - Realistic sensor physics (temperature, vibration, current)
  - OEE tracking (Availability, Quality)
  - Wear accumulation → rising fault probability over time
"""

from __future__ import annotations

import simpy
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_store import FactoryDataStore, AlarmEntry


# ──────────────────────────────────────────────
#  Enums & value objects
# ──────────────────────────────────────────────

class MachineState(Enum):
    IDLE        = "IDLE"
    PROCESSING  = "PROCESSING"
    FAULT       = "FAULT"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class SensorData:
    temperature: float    = 25.0   # °C
    vibration_rms: float  = 0.1    # mm/s
    motor_current: float  = 0.5    # A
    cycle_time_last: float = 0.0   # s


# ──────────────────────────────────────────────
#  Metric tracker (OEE components)
# ──────────────────────────────────────────────

class MachineMetrics:
    def __init__(self) -> None:
        self.parts_produced: int = 0
        self.parts_scrapped: int = 0
        self.fault_count: int = 0
        self._processing_time: float = 0.0
        self._idle_time: float = 0.0
        self._fault_time: float = 0.0

    def record_idle(self, dt: float) -> None:
        self._idle_time += dt

    def record_processing(self, dt: float) -> None:
        self._processing_time += dt

    def record_fault(self, dt: float) -> None:
        self._fault_time += dt
        self.fault_count += 1

    @property
    def availability(self) -> float:
        """Fraction of planned time the machine was NOT in fault. (0–1)"""
        total = self._processing_time + self._idle_time + self._fault_time
        if total == 0:
            return 1.0
        return (self._processing_time + self._idle_time) / total

    @property
    def quality(self) -> float:
        total = self.parts_produced + self.parts_scrapped
        if total == 0:
            return 1.0
        return self.parts_produced / total


# ──────────────────────────────────────────────
#  Machine SimPy process
# ──────────────────────────────────────────────

class Machine:
    """
    A single machine in the production line.
    Runs as a SimPy process that:
      1. Waits for a part in `in_buffer`
      2. Processes it (randomized cycle time)
      3. Possibly faults and self-repairs
      4. Applies quality check; passes good parts to `out_buffer`
    """

    def __init__(
        self,
        env: simpy.Environment,
        machine_id: str,
        station_id: str,
        station_name: str,
        config: dict,
        in_buffer: simpy.Store,
        out_buffer: Optional[simpy.Store],
        data_store: "FactoryDataStore",
        rng: np.random.Generator,
    ) -> None:
        self.env = env
        self.machine_id = machine_id
        self.station_id = station_id
        self.station_name = station_name
        self.config = config
        self.in_buffer = in_buffer
        self.out_buffer = out_buffer
        self._store = data_store
        self._rng = rng

        self.state = MachineState.IDLE
        self.sensors = SensorData()
        self.metrics = MachineMetrics()

        # Wear [0, 1]: increases each cycle and on faults; resets partially after repair
        self._wear: float = 0.0
        # Individual machine has slight ambient-temperature variation
        self._ambient_temp: float = 22.0 + rng.uniform(-3.0, 3.0)

        # Launch SimPy process
        self.process = env.process(self._run())

    # ── Internal helpers ──────────────────────────────────────────────────

    def _cycle_time(self) -> float:
        base = self.config["cycle_time_mean"]
        std = self.config["cycle_time_std"]
        # Wear increases mean cycle time by up to 15 %
        degraded = base * (1.0 + 0.15 * self._wear)
        return max(1.0, self._rng.normal(degraded, std))

    def _fault_occurs(self) -> bool:
        base_p = self.config["fault_probability"]
        # Wear triples the fault probability at wear=1
        p = base_p * (1.0 + 2.0 * self._wear)
        return bool(self._rng.random() < p)

    def _update_sensors(self, *, processing: bool, fault: bool) -> None:
        rng = self._rng
        if fault:
            self.sensors.temperature   = self._ambient_temp + rng.uniform(45.0, 85.0)
            self.sensors.vibration_rms = rng.uniform(6.0, 22.0)
            self.sensors.motor_current = rng.uniform(16.0, 28.0)
        elif processing:
            wear_vib = 1.0 + self._wear * 0.6
            self.sensors.temperature   = self._ambient_temp + rng.uniform(18.0, 35.0)
            self.sensors.vibration_rms = rng.uniform(0.8, 2.5) * wear_vib
            self.sensors.motor_current = rng.uniform(8.0, 14.0)
        else:  # idle
            self.sensors.temperature   = self._ambient_temp + rng.uniform(0.5, 4.0)
            self.sensors.vibration_rms = rng.uniform(0.05, 0.25)
            self.sensors.motor_current = rng.uniform(0.4, 1.8)

    def _emit_alarm(self, level: str, message: str) -> None:
        from core.data_store import AlarmEntry
        self._store.add_alarm(AlarmEntry(
            level=level,
            machine=self.machine_id,
            station=self.station_name,
            message=message,
            sim_time=self.env.now,
        ))

    # ── Main process ─────────────────────────────────────────────────────

    def _run(self):
        while True:
            # ── Wait for a part ──────────────────────────────────────────
            self.state = MachineState.IDLE
            self._update_sensors(processing=False, fault=False)
            idle_start = self.env.now

            part = yield self.in_buffer.get()

            self.metrics.record_idle(self.env.now - idle_start)

            # ── Process the part ─────────────────────────────────────────
            self.state = MachineState.PROCESSING
            ct = self._cycle_time()
            self.sensors.cycle_time_last = ct
            self._update_sensors(processing=True, fault=False)

            yield self.env.timeout(ct)
            self.metrics.record_processing(ct)

            # ── Fault check ──────────────────────────────────────────────
            if self._fault_occurs():
                self.state = MachineState.FAULT
                self._update_sensors(processing=False, fault=True)
                self._emit_alarm("CRITICAL", "Machine fault — initiating repair")

                mttr = max(30.0, self._rng.normal(
                    self.config["mttr_mean"], self.config["mttr_std"]
                ))
                fault_start = self.env.now
                yield self.env.timeout(mttr)
                self.metrics.record_fault(self.env.now - fault_start)

                # Wear increases on fault; partial reset after repair
                self._wear = min(1.0, self._wear + 0.08)
                self._wear = max(0.0, self._wear - 0.05)   # repair removes some wear

                self.state = MachineState.IDLE
                self._update_sensors(processing=False, fault=False)
                # Part is lost during a fault — loop back without forwarding
                self.metrics.parts_scrapped += 1
                continue

            # ── Quality gate ─────────────────────────────────────────────
            if self._rng.random() > self.config["quality_yield"]:
                self.metrics.parts_scrapped += 1
                self._wear = min(1.0, self._wear + 0.005)
            else:
                self.metrics.parts_produced += 1
                if self.out_buffer is not None:
                    yield self.out_buffer.put(part)

            # Small wear accumulation per cycle
            self._wear = min(1.0, self._wear + 0.0008)
