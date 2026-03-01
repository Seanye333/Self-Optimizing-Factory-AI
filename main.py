"""
main.py
Entry point for the Self-Optimizing Factory AI.

Usage
-----
# Run simulation headless (prints live stats to stdout)
    python main.py

# Launch the Streamlit dashboard (simulation starts automatically inside it)
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _print_stats(snapshot) -> None:
    """Pretty-print a factory snapshot to stdout."""
    sim_hm = f"{int(snapshot.sim_time // 3600)}h {int((snapshot.sim_time % 3600) // 60):02d}m"
    print(
        f"\r[Sim {sim_hm}] "
        f"OEE={snapshot.oee:.1f}%  "
        f"Out={snapshot.total_parts_produced:>6,}  "
        f"Scrap={snapshot.total_parts_scrapped:>4,}  "
        f"Alarms={len(store.get_alarms()):>3}  ",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    from core.data_store import FactoryDataStore
    from core.factory_env import FactoryEnvironment

    store = FactoryDataStore.instance()

    # ── Signal handler for clean Ctrl-C ──────────────────────────────────
    def _handle_sigint(sig, frame):
        snap = store.get_current()
        if snap:
            print(f"\n\nFinal state — sim time: {snap.sim_time / 3600:.2f}h")
            print(f"  Total output : {snap.total_parts_produced:,} parts")
            print(f"  Total scrapped: {snap.total_parts_scrapped:,} parts")
            print(f"  OEE          : {snap.oee:.1f}%")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    logger.info("Starting headless simulation (Ctrl-C to stop)")
    logger.info("To view the live dashboard run:  streamlit run dashboard/app.py")

    # ── Run simulation in foreground ─────────────────────────────────────
    env = FactoryEnvironment()

    # Print stats every real-second while simulation runs
    import threading

    def _stats_loop():
        while True:
            time.sleep(1.0)
            snap = store.get_current()
            if snap:
                _print_stats(snap)

    stats_thread = threading.Thread(target=_stats_loop, daemon=True)
    stats_thread.start()

    env.run()   # blocks forever (or until KeyboardInterrupt)
