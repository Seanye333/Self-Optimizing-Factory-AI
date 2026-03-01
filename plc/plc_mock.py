"""
plc/plc_mock.py
Generates realistic industrial PLC tag data from the live FactoryDataStore.

In a real deployment this module is replaced by an OPC-UA / Modbus / MQTT
client that reads from actual PLC hardware.  Here we synthesise the same
signals so the rest of the stack can be developed and tested offline.

PLC tag structure mirrors a typical Siemens S7 data block:
    DB1.DBX  → status bits
    DB1.DBD  → real (float) values
    DB1.DBW  → word (int) values
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from core.data_store import FactoryDataStore, MachineSnapshot


# ──────────────────────────────────────────────
#  PLC tag value objects
# ──────────────────────────────────────────────

@dataclass
class PLCTag:
    address: str        # e.g. "DB1.DBD0"
    name: str           # human-readable tag name
    value: float | int | bool
    unit: str = ""
    quality: str = "GOOD"   # GOOD | BAD | UNCERTAIN


@dataclass
class PLCDataPoint:
    """
    One complete PLC scan-cycle snapshot for a single machine.
    Mirrors what you'd read from a real PLC data block.
    """
    machine_id: str
    timestamp: float        # wall-clock epoch seconds
    sim_time: float         # simulation clock seconds
    tags: List[PLCTag] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float | int | bool | str]:
        return {t.name: t.value for t in self.tags}


# ──────────────────────────────────────────────
#  Signal physics helpers
# ──────────────────────────────────────────────

def _add_noise(value: float, noise_pct: float, rng: np.random.Generator) -> float:
    """Add Gaussian measurement noise (noise_pct = fraction of value)."""
    return value + rng.normal(0.0, abs(value) * noise_pct + 1e-6)


def _motor_current_with_harmonics(base_current: float, t: float, rng: np.random.Generator) -> float:
    """
    Simulate motor current with 50 Hz fundamental + 3rd/5th harmonics + noise.
    Real motor current analysis is the basis of motor current signature analysis (MCSA).
    """
    f = 50.0  # Hz
    i_fund  = base_current * math.sin(2 * math.pi * f * t)
    i_3rd   = 0.05 * base_current * math.sin(2 * math.pi * 3 * f * t)
    i_5th   = 0.02 * base_current * math.sin(2 * math.pi * 5 * f * t)
    noise   = rng.normal(0, 0.05)
    return abs(i_fund + i_3rd + i_5th) + noise


def _vibration_with_bearing_defect(base_vib: float, wear: float, t: float) -> float:
    """
    Simulate vibration with an optional bearing defect frequency component.
    Bearing defect frequency ≈ 3.5 × shaft RPM / 60 (simplified).
    """
    shaft_rpm = 1450
    bpfo = 3.5 * shaft_rpm / 60.0   # ball-pass frequency, outer race
    defect_amp = wear * 0.8          # defect grows with wear
    defect = defect_amp * abs(math.sin(2 * math.pi * bpfo * t))
    return base_vib + defect


# ──────────────────────────────────────────────
#  PLC Mock Generator
# ──────────────────────────────────────────────

class PLCMockGenerator:
    """
    Polls the FactoryDataStore and returns PLC-format tag lists
    for every machine in the current snapshot.

    Usage:
        gen = PLCMockGenerator()
        while True:
            points = gen.scan()   # list[PLCDataPoint]
            # publish via MQTT, write to InfluxDB, etc.
            time.sleep(0.1)
    """

    _BASE_DB = 10   # Siemens DB number

    def __init__(self, seed: int = 7) -> None:
        self._store = FactoryDataStore.instance()
        self._rng = np.random.default_rng(seed)
        self._t0 = time.time()

    def scan(self) -> List[PLCDataPoint]:
        """Return one PLCDataPoint per machine in the current snapshot."""
        snapshot = self._store.get_current()
        if snapshot is None:
            return []

        t_wall = time.time()
        t_rel = t_wall - self._t0

        points: List[PLCDataPoint] = []
        db_num = self._BASE_DB

        for station in snapshot.stations:
            for m in station.machines:
                points.append(self._machine_to_plc(m, t_rel, db_num, snapshot.sim_time))
                db_num += 1

        return points

    # ── Per-machine tag generation ─────────────────────────────────────

    def _machine_to_plc(
        self,
        m: MachineSnapshot,
        t_rel: float,
        db: int,
        sim_time: float,
    ) -> PLCDataPoint:
        rng = self._rng
        wear_frac = m.wear / 100.0

        # Derive physical signals from snapshot + physics models
        raw_temp     = _add_noise(m.temperature, 0.005, rng)
        raw_current  = _motor_current_with_harmonics(m.motor_current, t_rel, rng)
        raw_vib      = _vibration_with_bearing_defect(m.vibration_rms, wear_frac, t_rel)
        raw_vib      = _add_noise(raw_vib, 0.02, rng)

        # PLC status word: bit-encoded machine state
        state_bits = {
            "IDLE": 0b0001,
            "PROCESSING": 0b0010,
            "FAULT": 0b1000,
            "MAINTENANCE": 0b0100,
        }.get(m.state, 0b0000)

        tags = [
            PLCTag(f"DB{db}.DBX0.0", f"{m.machine_id}.Running",
                   m.state == "PROCESSING", quality="GOOD"),
            PLCTag(f"DB{db}.DBX0.1", f"{m.machine_id}.Fault",
                   m.state == "FAULT", quality="GOOD"),
            PLCTag(f"DB{db}.DBX0.2", f"{m.machine_id}.Maintenance",
                   m.state == "MAINTENANCE", quality="GOOD"),
            PLCTag(f"DB{db}.DBW2",   f"{m.machine_id}.StatusWord",
                   state_bits, unit="bits"),

            PLCTag(f"DB{db}.DBD4",   f"{m.machine_id}.Temperature",
                   round(raw_temp, 2), unit="°C"),
            PLCTag(f"DB{db}.DBD8",   f"{m.machine_id}.MotorCurrent",
                   round(raw_current, 3), unit="A"),
            PLCTag(f"DB{db}.DBD12",  f"{m.machine_id}.VibrationRMS",
                   round(raw_vib, 4), unit="mm/s"),
            PLCTag(f"DB{db}.DBD16",  f"{m.machine_id}.CycleTime",
                   round(m.cycle_time_last, 1), unit="s"),
            PLCTag(f"DB{db}.DBD20",  f"{m.machine_id}.WearIndex",
                   round(wear_frac, 4), unit=""),

            PLCTag(f"DB{db}.DBW24",  f"{m.machine_id}.PartsProduced",
                   m.parts_produced, unit="pcs"),
            PLCTag(f"DB{db}.DBW26",  f"{m.machine_id}.PartsScrapped",
                   m.parts_scrapped, unit="pcs"),
            PLCTag(f"DB{db}.DBW28",  f"{m.machine_id}.FaultCount",
                   m.fault_count, unit=""),

            PLCTag(f"DB{db}.DBD30",  f"{m.machine_id}.Availability",
                   round(m.availability, 1), unit="%"),
        ]

        return PLCDataPoint(
            machine_id=m.machine_id,
            timestamp=time.time(),
            sim_time=sim_time,
            tags=tags,
        )

    # ── Convenience: flat dict per machine ────────────────────────────

    def scan_flat(self) -> List[dict]:
        """Return scan results as plain dicts — easy to push to InfluxDB/Kafka."""
        return [
            {"machine_id": p.machine_id, "timestamp": p.timestamp, **p.as_dict()}
            for p in self.scan()
        ]
