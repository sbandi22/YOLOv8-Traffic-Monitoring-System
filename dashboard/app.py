"""Streamlit analytics dashboard for the YOLOv8 Traffic Monitoring System.

Connects to the FastAPI backend, polls the analytics endpoints, and renders
live metrics, charts, the cumulative heatmap, and congestion hotspots."""
from __future__ import annotations

import os
import time
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from components import metric_card, render_hotspots_table, traffic_chart

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH", "2"))

st.set_page_config(
    page_title="YOLOv8 Traffic Monitoring",
    page_icon="\U0001F6A6",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_get(path, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        st.error(f"API error on {path}: {e}")
        return None


def render_sidebar():
    with st.sidebar:
        st.title("\U0001F6A6 Traffic Monitor")
        st.markdown("---")
        st.text_input("API Base URL", value=API_BASE, key="api_base")
        st.slider("Refresh interval (s)", 1, 10, REFRESH_SECONDS, key="refresh")
        st.markdown("---")
        h = safe_get("/health")
        if h is not None:
            data = h.json()
            st.success(f"Backend: {data.get('status')}")
            st.caption(f"v{data.get('version')}  -  device: {data.get('device')}")
        else:
            st.error("Backend offline")


def render_main():
    st.title("\U0001F4CA Real-time Traffic Analytics")

    placeholder = st.empty()
    while True:
        summary = safe_get("/analytics/summary")
        if summary is None:
            time.sleep(st.session_state.get("refresh", REFRESH_SECONDS))
            continue
        data = summary.json()

        with placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Frames processed", data.get("frames", 0))
            with c2: metric_card("Avg density", f"{data.get('avg_density', 0):.2%}")
            with c3: metric_card("Avg speed (km/h)", f"{data.get('avg_speed', 0):.1f}")
            with c4: metric_card("Peak congestion", f"{data.get('peak_congestion', 0):.2f}")

            colA, colB = st.columns([2, 1])
            with colA:
                st.subheader("\U0001F525 Heatmap")
                hm = safe_get("/analytics/heatmap")
                if hm is not None:
                    st.image(BytesIO(hm.content), use_column_width=True)

            with colB:
                st.subheader("\U0001F4CD Congestion Hotspots")
                hs = safe_get("/congestion/hotspots")
                if hs is not None:
                    render_hotspots_table(hs.json())

            st.subheader("\U0001F6E3 Lanes")
            lanes = data.get("lanes", {})
            if lanes:
                df = pd.DataFrame(
                    [{"lane": k, "count": v.get("count", 0), "avg_speed": v.get("avg_speed", 0)} for k, v in lanes.items()]
                )
                st.dataframe(df, use_container_width=True)

            st.subheader("\U0001F3AF Vehicle Counts (line crossings)")
            counts = data.get("final_counts", {})
            if counts:
                df = pd.DataFrame(
                    [{"line": k, "in": v.get("in", 0), "out": v.get("out", 0)} for k, v in counts.items()]
                )
                st.dataframe(df, use_container_width=True)
                traffic_chart(df)

        time.sleep(int(st.session_state.get("refresh", REFRESH_SECONDS)))


def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
