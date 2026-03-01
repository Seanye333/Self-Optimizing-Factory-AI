"""
dashboard/app.py
Streamlit live command-centre for the Factory Digital Twin.

Run with:
    streamlit run dashboard/app.py

The simulation starts automatically in a background thread the first
time the dashboard is loaded.  Subsequent Streamlit reruns (triggered
by st.rerun()) share the same process and the same FactoryDataStore
singleton — no IPC needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Make project root importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_store import FactoryDataStore, FactorySnapshot, StationSnapshot

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Factory AI — Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Remove default top padding */
  .block-container { padding-top: 1rem; }

  /* Machine status cards */
  .mcard {
    border-radius: 8px;
    padding: 10px 12px;
    margin: 3px 0;
    font-size: 13px;
    line-height: 1.5;
    text-align: center;
  }
  .mcard-processing  { background: #145a32; color: #d5f5e3; }
  .mcard-idle        { background: #1a4a7a; color: #d6eaf8; }
  .mcard-fault       { background: #7b241c; color: #fadbd8; }
  .mcard-maintenance { background: #6e2f00; color: #fdebd0; }
  .mcard-unknown     { background: #333;    color: #ccc;    }

  /* KPI tiles */
  div[data-testid="metric-container"] {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Bootstrap simulation
# ──────────────────────────────────────────────

store = FactoryDataStore.instance()
store.ensure_simulation_running()

# ──────────────────────────────────────────────
#  Wait for first snapshot
# ──────────────────────────────────────────────

snapshot: FactorySnapshot | None = store.get_current()

if snapshot is None:
    st.title("🏭 Factory AI — Digital Twin")
    with st.spinner("⏳ Simulation starting — waiting for first data..."):
        for _ in range(30):          # wait up to 30 s
            time.sleep(1)
            snapshot = store.get_current()
            if snapshot is not None:
                break
    if snapshot is None:
        st.error("Simulation did not produce data within 30 seconds. Check logs.")
        st.stop()
    st.rerun()

# ──────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────

sim_h  = snapshot.sim_time / 3600.0
sim_hm = f"{int(sim_h)}h {int((sim_h % 1) * 60):02d}m"

header_l, header_r = st.columns([3, 1])
with header_l:
    st.markdown("## 🏭 Self-Optimizing Factory AI")
    st.caption(f"Phase 1 — Digital Twin · Sim time: **{sim_hm}** · "
               f"Wall clock: **{time.strftime('%H:%M:%S')}**")
with header_r:
    oee_color = "#27ae60" if snapshot.oee >= 75 else ("#e67e22" if snapshot.oee >= 55 else "#e74c3c")
    st.markdown(
        f"<div style='text-align:right;font-size:2.4rem;font-weight:700;color:{oee_color}'>"
        f"OEE {snapshot.oee:.1f}%</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ──────────────────────────────────────────────
#  Top KPI row
# ──────────────────────────────────────────────

total_all = snapshot.total_parts_produced + snapshot.total_parts_scrapped
scrap_rate = (snapshot.total_parts_scrapped / total_all * 100) if total_all else 0.0
alarms = store.get_alarms()
critical_count = sum(1 for a in alarms if a.level == "CRITICAL")
active_machines = sum(
    1 for st_snap in snapshot.stations
    for m in st_snap.machines
    if m.state == "PROCESSING"
)
faulted = sum(
    1 for st_snap in snapshot.stations
    for m in st_snap.machines
    if m.state == "FAULT"
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Output",     f"{snapshot.total_parts_produced:,}")
k2.metric("Scrapped",         f"{snapshot.total_parts_scrapped:,}")
k3.metric("Scrap Rate",       f"{scrap_rate:.1f}%")
k4.metric("Active Machines",  f"{active_machines}")
k5.metric("Machines Faulted", f"{faulted}", delta=None)
k6.metric("Critical Alarms",  f"{critical_count}",
          delta=None if critical_count == 0 else f"+{critical_count}")

st.divider()

# ──────────────────────────────────────────────
#  Station + Machine Status Grid
# ──────────────────────────────────────────────

st.subheader("Production Line — Live Status")

state_css = {
    "PROCESSING":  "mcard mcard-processing",
    "IDLE":        "mcard mcard-idle",
    "FAULT":       "mcard mcard-fault",
    "MAINTENANCE": "mcard mcard-maintenance",
}
state_icon = {
    "PROCESSING":  "⚙️",
    "IDLE":        "💤",
    "FAULT":       "🔴",
    "MAINTENANCE": "🔧",
}

cols = st.columns(len(snapshot.stations))
for col, station in zip(cols, snapshot.stations):
    with col:
        st.markdown(f"**{station.name}**")

        # Buffer bar
        buf_pct = station.buffer_level / station.buffer_capacity if station.buffer_capacity else 0
        buf_label = f"Buffer {station.buffer_level}/{station.buffer_capacity}"
        buf_color = "normal" if buf_pct < 0.7 else ("off" if buf_pct < 0.9 else "inverse")
        st.progress(min(buf_pct, 1.0), text=buf_label)

        # Machine cards
        for m in station.machines:
            css   = state_css.get(m.state, "mcard mcard-unknown")
            icon  = state_icon.get(m.state, "❓")
            avail = m.availability
            wear  = m.wear

            st.markdown(
                f'<div class="{css}">'
                f'{icon} <b>{m.machine_id}</b><br>'
                f'{m.state}<br>'
                f'<small>'
                f'🌡 {m.temperature}°C &nbsp; '
                f'⚡ {m.motor_current}A &nbsp; '
                f'〰 {m.vibration_rms} mm/s<br>'
                f'Parts: {m.parts_produced} &nbsp; Faults: {m.fault_count}<br>'
                f'Avail: {avail:.1f}% &nbsp; Wear: {wear:.1f}%'
                f'</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.caption(f"Good: {station.parts_produced} | Scrap: {station.parts_scrapped}")

st.divider()

# ──────────────────────────────────────────────
#  Charts
# ──────────────────────────────────────────────

history = store.get_history()
has_history = len(history) > 2

chart_l, chart_r = st.columns(2)

# ── Cumulative throughput ───────────────────────────────────────────────
with chart_l:
    st.subheader("Throughput")
    if has_history:
        df_tp = pd.DataFrame([
            {"Sim Time (h)": h.sim_time / 3600, "Good Parts": h.total_parts_produced,
             "Scrapped": h.total_parts_scrapped}
            for h in history
        ])
        fig = px.line(
            df_tp, x="Sim Time (h)", y=["Good Parts", "Scrapped"],
            color_discrete_map={"Good Parts": "#27ae60", "Scrapped": "#e74c3c"},
        )
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collecting data...")

# ── Buffer WIP levels ──────────────────────────────────────────────────
with chart_r:
    st.subheader("Buffer WIP Levels")
    if has_history:
        station_names = [s.name for s in snapshot.stations]
        df_buf = pd.DataFrame([
            {"Sim Time (h)": h.sim_time / 3600,
             **{s.name: s.buffer_level for s in h.stations}}
            for h in history
        ])
        fig2 = px.line(df_buf, x="Sim Time (h)", y=station_names)
        fig2.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Collecting data...")

# ── OEE trend + Machine availability ─────────────────────────────────
oee_l, oee_r = st.columns(2)

with oee_l:
    st.subheader("OEE Trend")
    if has_history:
        df_oee = pd.DataFrame([
            {"Sim Time (h)": h.sim_time / 3600, "OEE (%)": h.oee}
            for h in history
        ])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_oee["Sim Time (h)"], y=df_oee["OEE (%)"],
            fill="tozeroy", mode="lines",
            line=dict(color="#3498db", width=2),
            fillcolor="rgba(52,152,219,0.2)",
        ))
        fig3.add_hline(y=75, line_dash="dot", line_color="green",
                       annotation_text="Target 75%")
        fig3.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                           yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Collecting data...")

with oee_r:
    st.subheader("Machine Availability vs Wear")
    machine_rows = [
        {"Machine": m.machine_id,
         "Availability (%)": m.availability,
         "Wear (%)": m.wear}
        for st_snap in snapshot.stations
        for m in st_snap.machines
    ]
    if machine_rows:
        df_m = pd.DataFrame(machine_rows)
        fig4 = px.bar(
            df_m, x="Machine", y="Availability (%)",
            color="Wear (%)", color_continuous_scale="RdYlGn_r",
            range_y=[0, 100],
        )
        fig4.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────
#  Sensor deep-dive (per-machine sparklines)
# ──────────────────────────────────────────────

with st.expander("Sensor Detail — Temperature & Vibration History", expanded=False):
    if has_history:
        # Build per-machine time-series from history
        machine_ids = [
            m.machine_id
            for st_snap in snapshot.stations
            for m in st_snap.machines
        ]
        records = []
        for h in history:
            for st_snap in h.stations:
                for m in st_snap.machines:
                    records.append({
                        "Sim Time (h)": h.sim_time / 3600,
                        "Machine": m.machine_id,
                        "Temperature (°C)": m.temperature,
                        "Vibration (mm/s)": m.vibration_rms,
                    })
        df_sens = pd.DataFrame(records)

        s1, s2 = st.columns(2)
        with s1:
            fig5 = px.line(df_sens, x="Sim Time (h)", y="Temperature (°C)",
                           color="Machine", title="Temperature")
            fig5.update_layout(height=300, margin=dict(t=30))
            st.plotly_chart(fig5, use_container_width=True)
        with s2:
            fig6 = px.line(df_sens, x="Sim Time (h)", y="Vibration (mm/s)",
                           color="Machine", title="Vibration RMS")
            fig6.update_layout(height=300, margin=dict(t=30))
            st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Need more history — check back shortly.")

# ──────────────────────────────────────────────
#  Alarm log
# ──────────────────────────────────────────────

st.subheader("Alarm Log")

recent_alarms = store.get_alarms()[-30:]
if recent_alarms:
    alarm_df = pd.DataFrame([
        {
            "Level":   a.level,
            "Machine": a.machine,
            "Station": a.station,
            "Message": a.message,
            "Sim Time": f"{a.sim_time / 3600:.2f}h",
        }
        for a in reversed(recent_alarms)
    ])

    def _highlight(row):
        color = {"CRITICAL": "background-color:#5c1010;color:#fff",
                 "WARNING":  "background-color:#5c3d00;color:#fff"}.get(row["Level"], "")
        return [color] * len(row)

    st.dataframe(
        alarm_df.style.apply(_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No alarms — all systems nominal.")

# ──────────────────────────────────────────────
#  Sidebar — controls
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    st.caption("Phase 1 — Read only.  Write-back in Phase 4.")

    st.metric("Simulation Speed", "60× real-time")
    st.metric("Publish interval",  "5 sim-seconds")

    st.divider()
    st.markdown("**Next phases:**")
    st.markdown("- Phase 2: Predictive Maintenance AI")
    st.markdown("- Phase 3: RL Production Optimizer")
    st.markdown("- Phase 4: Closed-loop PLC Write-back")

# ──────────────────────────────────────────────
#  Auto-refresh (1 real-second)
# ──────────────────────────────────────────────

time.sleep(1.0)
st.rerun()
