import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Safety Monitor", page_icon="🦺", layout="wide")

ALERT_LOG = Path("logs/alerts.csv")

@st.cache_data(ttl=3)
def load_alerts():
    if not ALERT_LOG.exists():
        return pd.DataFrame(columns=["timestamp_str","severity","event_type","message","frame_id"])
    try:
        df = pd.read_csv(ALERT_LOG, parse_dates=["timestamp_str"])
        df.rename(columns={"timestamp_str": "timestamp"}, inplace=True)
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

with st.sidebar:
    st.title("Safety Monitor")
    auto_refresh = st.checkbox("Auto-refresh (3s)", value=True)
    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

st.title("Industrial Safety Monitoring Dashboard")

df = load_alerts()
n_total    = len(df)
n_critical = len(df[df["severity"] == "CRITICAL"]) if not df.empty else 0
n_warning  = len(df[df["severity"] == "WARNING"])  if not df.empty else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Alerts", n_total)
col2.metric("Critical",     n_critical)
col3.metric("Warnings",     n_warning)

st.divider()

if not df.empty and "timestamp" in df.columns:
    st.subheader("Alert Timeline")
    timeline = (df.set_index("timestamp").resample("1min")["severity"]
                  .count().reset_index().rename(columns={"severity":"count"}))
    fig = px.bar(timeline, x="timestamp", y="count")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Alerts")
if not df.empty:
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
else:
    st.info("No alerts yet. Start main.py to begin.")

if auto_refresh:
    time.sleep(3)
    st.rerun()