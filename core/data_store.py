"""
core/data_store.py
Thread-safe singleton that holds all live factory state.
Both the SimPy simulation thread and the Streamlit dashboard
import this module and share the same FactoryDataStore instance.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


# ──────────────────────────────────────────────
#  Data-transfer objects (snapshot types)
# ──────────────────────────────────────────────

@dataclass
class MachineSnapshot:
    machine_id: str
    station_id: str
    state: str                  # IDLE | PROCESSING | FAULT | MAINTENANCE
    temperature: float          # °C
    vibration_rms: float        # mm/s
    motor_current: float        # A
    cycle_time_last: float      # seconds
    parts_produced: int
    parts_scrapped: int
    fault_count: int
    availability: float         # 0-100 %
    wear: float                 # 0-100 %  (increases with cycles / faults)
    sim_time: float             # simulation clock (seconds)


@dataclass
class StationSnapshot:
    station_id: str
    name: str
    buffer_level: int           # items currently in the downstream buffer
    buffer_capacity: int
    machines: List[MachineSnapshot]
    parts_produced: int         # cumulative good parts through this station
    parts_scrapped: int


@dataclass
class FactorySnapshot:
    sim_time: float
    real_time: float
    stations: List[StationSnapshot]
    total_parts_produced: int
    total_parts_scrapped: int
    oee: float                  # 0-100 %


@dataclass
class AlarmEntry:
    level: str          # CRITICAL | WARNING | INFO
    machine: str
    station: str
    message: str
    sim_time: float
    real_time: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
#  Thread-safe singleton
# ──────────────────────────────────────────────

_MAX_SNAPSHOT_HISTORY = 600   # ~10 minutes at 1 snapshot/s
_MAX_ALARMS = 200


class FactoryDataStore:
    """
    Singleton shared between the simulation thread and Streamlit.
    All public methods are thread-safe.
    """

    _instance: Optional["FactoryDataStore"] = None
    _class_lock = threading.Lock()

    # ── singleton access ──
    @classmethod
    def instance(cls) -> "FactoryDataStore":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls.__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._rw_lock = threading.RLock()
        self.current_snapshot: Optional[FactorySnapshot] = None
        self.snapshot_history: Deque[FactorySnapshot] = deque(maxlen=_MAX_SNAPSHOT_HISTORY)
        self.alarms: Deque[AlarmEntry] = deque(maxlen=_MAX_ALARMS)
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_started = False

    # ── write (called from simulation thread) ──
    def update(self, snapshot: FactorySnapshot) -> None:
        with self._rw_lock:
            self.current_snapshot = snapshot
            self.snapshot_history.append(snapshot)

    def add_alarm(self, alarm: AlarmEntry) -> None:
        with self._rw_lock:
            self.alarms.append(alarm)

    # ── read (called from Streamlit thread) ──
    def get_current(self) -> Optional[FactorySnapshot]:
        with self._rw_lock:
            return self.current_snapshot

    def get_history(self) -> List[FactorySnapshot]:
        with self._rw_lock:
            return list(self.snapshot_history)

    def get_alarms(self) -> List[AlarmEntry]:
        with self._rw_lock:
            return list(self.alarms)

    # ── simulation lifecycle ──
    def ensure_simulation_running(self) -> None:
        """Start the factory simulation in a background daemon thread (once only)."""
        with self._class_lock:
            alive = self._sim_thread is not None and self._sim_thread.is_alive()
            if self._sim_started and alive:
                return
            self._sim_started = True

        def _run():
            # Import here to avoid circular imports at module level
            from core.factory_env import FactoryEnvironment
            try:
                env = FactoryEnvironment()
                env.run()
            except Exception as exc:
                import traceback
                print(f"[SimThread] FATAL: {exc}")
                traceback.print_exc()

        t = threading.Thread(target=_run, name="SimThread", daemon=True)
        with self._class_lock:
            self._sim_thread = t
        t.start()
