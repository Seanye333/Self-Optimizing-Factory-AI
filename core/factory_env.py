"""
core/factory_env.py
Sets up the full production line as a SimPy simulation and drives
the publish loop that feeds the shared FactoryDataStore.

Architecture:
    Parts arrive (Poisson) → Buffer[0]
    → Station1 machines   → Buffer[1]
    → Station2 machines   → Buffer[2]
    → ...
    → StationN machines   → Buffer[N]  (output sink)

The simulation runs via simpy.rt.RealtimeEnvironment so that
  1 real-second ≈ time_scale simulation-seconds.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import simpy
import simpy.rt
import yaml

from core.data_store import (
    AlarmEntry,
    FactoryDataStore,
    FactorySnapshot,
    MachineSnapshot,
    StationSnapshot,
)
from core.machines import Machine

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "factory_config.yaml"


class FactoryEnvironment:
    """
    Wires up the SimPy production line and runs it in real-time.
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self._cfg = cfg["factory"]
        self._store = FactoryDataStore.instance()
        self._rng = np.random.default_rng(seed=42)

        self._time_scale: float = self._cfg["time_scale"]
        self._publish_interval: float = self._cfg["publish_interval"]

        # simpy.rt.RealtimeEnvironment: factor = real_seconds_per_sim_second
        # factor = 1/time_scale  → time_scale sim-seconds per real-second
        self._env = simpy.rt.RealtimeEnvironment(
            factor=1.0 / self._time_scale,
            strict=False,   # don't raise if sim falls slightly behind wall clock
        )

        self._machines: List[Machine] = []
        self._buffers: List[simpy.Store] = []
        self._machines_by_station: dict[str, List[Machine]] = {}

        self._setup_line()
        self._env.process(self._part_arrivals())
        self._env.process(self._publish_loop())

    # ── Line construction ─────────────────────────────────────────────────

    def _setup_line(self) -> None:
        stations = self._cfg["stations"]

        # Create one buffer per inter-station gap + output sink
        for i, scfg in enumerate(stations):
            cap = scfg["buffer_capacity"]
            self._buffers.append(simpy.Store(self._env, capacity=cap))
        # Output sink — effectively unlimited
        self._buffers.append(simpy.Store(self._env, capacity=99_999))

        # Create machines for every station
        for i, scfg in enumerate(stations):
            sid = scfg["id"]
            self._machines_by_station[sid] = []

            for m_idx in range(scfg["num_machines"]):
                machine_id = f"{sid}-M{m_idx + 1:02d}"
                machine = Machine(
                    env=self._env,
                    machine_id=machine_id,
                    station_id=sid,
                    station_name=scfg["name"],
                    config=scfg,
                    in_buffer=self._buffers[i],
                    out_buffer=self._buffers[i + 1],
                    data_store=self._store,
                    rng=np.random.default_rng(self._rng.integers(0, 2**31)),
                )
                self._machines.append(machine)
                self._machines_by_station[sid].append(machine)

        logger.info(
            "Line '%s' built: %d stations, %d machines",
            self._cfg["name"],
            len(stations),
            len(self._machines),
        )

    # ── Part-arrival generator ────────────────────────────────────────────

    def _part_arrivals(self):
        """Poisson part arrivals into the first buffer."""
        mean_ia = self._cfg["arrival"]["mean_interarrival"]
        part_id = 0
        while True:
            ia = self._rng.exponential(mean_ia)
            yield self._env.timeout(ia)
            # Non-blocking put: drop parts if buffer is full (represents supplier backlog)
            if len(self._buffers[0].items) < self._buffers[0].capacity:
                yield self._buffers[0].put({"id": part_id, "t_arrive": self._env.now})
            part_id += 1

    # ── State publisher ───────────────────────────────────────────────────

    def _publish_loop(self):
        """Emit a FactorySnapshot to the data store every publish_interval sim-seconds."""
        while True:
            yield self._env.timeout(self._publish_interval)
            self._publish()

    def _publish(self) -> None:
        stations_cfg = self._cfg["stations"]
        station_snapshots: List[StationSnapshot] = []
        total_produced = 0
        total_scrapped = 0

        for i, scfg in enumerate(stations_cfg):
            sid = scfg["id"]
            station_machines = self._machines_by_station.get(sid, [])

            machine_snaps: List[MachineSnapshot] = []
            st_produced = 0
            st_scrapped = 0

            for m in station_machines:
                machine_snaps.append(MachineSnapshot(
                    machine_id=m.machine_id,
                    station_id=m.station_id,
                    state=m.state.value,
                    temperature=round(m.sensors.temperature, 1),
                    vibration_rms=round(m.sensors.vibration_rms, 3),
                    motor_current=round(m.sensors.motor_current, 2),
                    cycle_time_last=round(m.sensors.cycle_time_last, 1),
                    parts_produced=m.metrics.parts_produced,
                    parts_scrapped=m.metrics.parts_scrapped,
                    fault_count=m.metrics.fault_count,
                    availability=round(m.metrics.availability * 100.0, 1),
                    wear=round(m._wear * 100.0, 1),
                    sim_time=self._env.now,
                ))
                st_produced += m.metrics.parts_produced
                st_scrapped += m.metrics.parts_scrapped

            # Downstream buffer (output of this station)
            downstream_buf = self._buffers[i + 1]
            buf_level = len(downstream_buf.items)

            station_snapshots.append(StationSnapshot(
                station_id=sid,
                name=scfg["name"],
                buffer_level=buf_level,
                buffer_capacity=scfg["buffer_capacity"],
                machines=machine_snaps,
                parts_produced=st_produced,
                parts_scrapped=st_scrapped,
            ))
            total_produced += st_produced
            total_scrapped += st_scrapped

        # ── Simplified OEE ──────────────────────────────────────────────
        # OEE = Availability × Quality  (Performance added in Phase 3)
        availabilities = [m.metrics.availability for m in self._machines]
        avg_avail = float(np.mean(availabilities)) if availabilities else 1.0

        total_all = total_produced + total_scrapped
        quality = total_produced / total_all if total_all > 0 else 1.0

        oee = round(avg_avail * quality * 100.0, 1)

        snapshot = FactorySnapshot(
            sim_time=self._env.now,
            real_time=time.time(),
            stations=station_snapshots,
            total_parts_produced=total_produced,
            total_parts_scrapped=total_scrapped,
            oee=oee,
        )
        self._store.update(snapshot)

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self, until: float | None = None) -> None:
        """
        Block and run the simulation.
        `until` = stop after N simulation-seconds (default: run forever).
        """
        logger.info(
            "Starting '%s' at %.0fx speed (1 real-s = %.0f sim-s)",
            self._cfg["name"],
            self._time_scale,
            self._time_scale,
        )
        try:
            if until is not None:
                self._env.run(until=until)
            else:
                self._env.run()
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user.")
        except Exception as exc:
            logger.exception("Simulation crashed: %s", exc)
            raise
