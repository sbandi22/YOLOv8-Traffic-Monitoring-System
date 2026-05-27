"""Reusable Streamlit UI components for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def metric_card(label: str, value):
    st.metric(label=label, value=value)


def render_hotspots_table(hotspots):
    if not hotspots:
        st.info("No congestion hotspots detected.")
        return
    df = pd.DataFrame(hotspots)
    st.dataframe(df, use_container_width=True)


def traffic_chart(df: pd.DataFrame):
    if df.empty:
        return
    melted = df.melt(id_vars=["line"], value_vars=["in", "out"], var_name="direction", value_name="count")
    fig = px.bar(melted, x="line", y="count", color="direction", barmode="group", title="Vehicle Counts per Line")
    st.plotly_chart(fig, use_container_width=True)
