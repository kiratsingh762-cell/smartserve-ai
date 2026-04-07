import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

API_BASE = "http://localhost:8000"

def fetch_summary():
    try:
        r = requests.get(f"{API_BASE}/analytics/summary", timeout=10)
        return r.json()
    except Exception as e:
        st.error(f"Could not fetch analytics: {e}")
        return None

def fetch_recent(limit=20):
    try:
        r = requests.get(f"{API_BASE}/analytics/recent?limit={limit}", timeout=10)
        return r.json().get("interactions", [])
    except:
        return []

def render_dashboard():
    st.set_page_config(page_title="SmartServe Analytics", layout="wide")

    st.title("SmartServe AI — Analytics Dashboard")
    st.caption("Live insights from customer interactions · Auto-refreshes every 30 seconds")

    if st.button("Refresh Now"):
        st.rerun()

    summary = fetch_summary()
    if not summary:
        st.warning("No analytics data yet. Send some messages in the chatbot first!")
        return

    # ── KPI Cards ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Total Interactions",
        value=summary["total_interactions"]
    )
    col2.metric(
        label="Unique Sessions",
        value=summary["unique_sessions"]
    )
    col3.metric(
        label="Escalation Rate",
        value=f"{summary['escalation_rate']}%",
        delta=f"{summary['total_escalations']} escalated",
        delta_color="inverse"
    )
    col4.metric(
        label="Auto-Resolved",
        value=f"{100 - summary['escalation_rate']}%",
        delta="by AI"
    )
    col5.metric(
        label="Avg Response Time",
        value=f"{summary['avg_response_ms']}ms",
        delta=f"{round(summary['avg_response_ms']/1000, 1)}s"
    )

    # ── Charts Row 1 ─────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Knowledge Source Usage")
        sources = summary.get("top_sources", [])
        if sources:
            df_src = pd.DataFrame(sources)
            df_src = df_src.sort_values("count", ascending=False)
            fig = px.pie(
                df_src, names="source", values="count",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No source data yet.")

    with col_b:
        st.subheader("Resolution Type")
        total = summary["total_interactions"]
        escalated = summary["total_escalations"]
        resolved  = total - escalated
        fig2 = px.pie(
            names=["Auto-Resolved by AI", "Escalated to Human"],
            values=[resolved, escalated],
            color_discrete_sequence=["#01696f", "#a12c7b"]
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────────────
    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Top Customer Questions")
        questions = summary.get("top_questions", [])
        if questions:
            df_q = pd.DataFrame(questions)
            df_q["question"] = df_q["question"].apply(
                lambda x: x[:45] + "..." if len(x) > 45 else x
            )
            df_q = df_q.sort_values("count")
            fig3 = go.Figure(go.Bar(
                x=df_q["count"], y=df_q["question"],
                orientation="h",
                marker_color="#01696f",
                text=df_q["count"], textposition="outside"
            ))
            fig3.update_layout(
                margin=dict(t=20, b=20, l=20, r=60),
                xaxis_title="Count",
                yaxis_title=""
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No question data yet.")

    with col_d:
        st.subheader("Response Confidence Levels")
        conf = summary.get("confidence_counts", {})
        if conf:
            colors_map = {"high": "#01696f", "medium": "#d19900", "low": "#a12c7b"}
            labels = list(conf.keys())
            values = list(conf.values())
            bar_colors = [colors_map.get(l, "#888") for l in labels]
            fig4 = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=bar_colors,
                text=values, textposition="outside"
            ))
            fig4.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis_title="Confidence Level",
                yaxis_title="Count",
                yaxis=dict(range=[0, max(values) + 1])
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No confidence data yet.")

    # ── Recent Interactions Table ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Recent Interactions")
    recent = fetch_recent(20)
    if recent:
        df_recent = pd.DataFrame(recent)
        df_recent["escalated"] = df_recent["escalated"].apply(
            lambda x: "Yes" if x else "No"
        )
        df_recent["response_ms"] = df_recent["response_ms"].apply(
            lambda x: f"{x}ms"
        )
        df_recent["timestamp"] = pd.to_datetime(
            df_recent["timestamp"]
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_recent = df_recent.rename(columns={
            "id":          "ID",
            "timestamp":   "Time (UTC)",
            "session_id":  "Session",
            "question":    "Question",
            "confidence":  "Confidence",
            "escalated":   "Escalated",
            "response_ms": "Response Time",
            "sources":     "Sources"
        })
        cols_show = ["ID","Time (UTC)","Session","Question",
                     "Confidence","Escalated","Response Time","Sources"]
        st.dataframe(
            df_recent[cols_show],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent interactions to display.")

    # ── Auto-refresh every 30 seconds ────────────────────────────────────
    st.markdown("---")
    st.caption("Dashboard auto-refreshes every 30 seconds")
    st.markdown(
        """<meta http-equiv="refresh" content="30">""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    render_dashboard()
