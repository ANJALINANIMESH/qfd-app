import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="manf_opt (anjnim)",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-theme CSS ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main background */
        .stApp { background-color: #0e1117; color: #e0e0e0; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

        /* Slider accent */
        .stSlider > div > div > div > div { background: #58a6ff !important; }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 12px 18px;
        }
        div[data-testid="metric-container"] label { color: #8b949e !important; }
        div[data-testid="metric-container"] div   { color: #58a6ff !important; }

        /* Section headers */
        h1, h2, h3 { color: #58a6ff !important; }

        /* Divider */
        hr { border-color: #30363d; }

        /* Tab styling */
        button[data-baseweb="tab"] { color: #8b949e !important; }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #58a6ff !important;
            border-bottom: 2px solid #58a6ff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CUSTOMER_REQS = [
    "Durability",
    "Low Weight",
    "Low Cost",
    "Safety",
    "Fatigue Life",
    "Corrosion Resistance",
    "Manufacturability",
]

TECH_DESCRIPTORS = [
    "Yield Strength",
    "Density",
    "Fatigue Strength",
    "Section Stiffness",
    "Fillet Radius",
    "Tolerance Capability",
    "Tooling Cost",
    "Cycle Time",
]

# QFD Relationship matrix  (rows = customer reqs, cols = tech descriptors)
# Values: 9 = strong, 3 = moderate, 1 = weak, 0 = none
QFD_MATRIX = np.array([
#   YS Den FS SS FR TC TIC CT
    [9, 0, 9, 9, 3, 1, 0, 0],   # Durability
    [0, 9, 0, 3, 0, 0, 0, 3],   # Low Weight
    [0, 0, 0, 0, 0, 3, 9, 9],   # Low Cost
    [9, 0, 3, 9, 9, 3, 0, 0],   # Safety
    [3, 0, 9, 3, 9, 0, 0, 0],   # Fatigue Life
    [3, 0, 1, 0, 3, 9, 0, 0],   # Corrosion Resistance
    [0, 3, 0, 0, 1, 9, 3, 9],   # Manufacturability
], dtype=float)


# Manufacturing process capability matrix
# Rows = processes
# Cols = technical descriptors
# Score: 0 to 10

PROCESS_SCORES = np.array([
#   YS Den FS SS FR TC TIC CT
    [10, 8, 9, 9, 9, 8, 8, 9],   # Forging
    [7,  7, 6, 7, 6, 7, 8, 8],   # Casting
    [6,  6, 5, 6, 5, 8, 7, 7],   # Welding / Fabrication
    [7,  5, 4, 7, 7, 4, 4, 3],   # CNC Machining
], dtype=float)

PROCESSES = ["Forging", "Casting", "Welding", "CNC Machining"]

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#58a6ff"
GRID_COLOR = "#30363d"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Customer Priorities")
    st.markdown("Adjust importance weights (1 = low, 10 = high)")
    st.markdown("---")

    weights = {}
    icons = ["", "", "", "", "", "", ""]
    for req, icon in zip(CUSTOMER_REQS, icons):
        weights[req] = st.slider(
            f"{icon} {req}",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key=req,
        )

    st.markdown("---")
    st.markdown("### Relationship Scale")
    st.markdown(
        """
        | Symbol | Value | Meaning |
        |--------|-------|---------|
        | ◉ | 9 | Strong |
        | ○ | 3 | Moderate |
        | △ | 1 | Weak |
        | — | 0 | None |
        """
    )

# ── Calculations ──────────────────────────────────────────────────────────────
weight_vector = np.array([weights[r] for r in CUSTOMER_REQS], dtype=float)

# Technical importance = sum(weight_i * relationship_ij) for each descriptor j
tech_importance = QFD_MATRIX.T @ weight_vector  # shape (8,)

# Normalise to 0-100
tech_importance_norm = 100 * tech_importance / tech_importance.max()

# Process suitability = dot(tech_importance_norm, process_scores_j) for each process
process_suitability = PROCESS_SCORES @ tech_importance_norm  # shape (4,)
process_suitability_norm = 100 * process_suitability / process_suitability.max()

# Ranked descriptors
rank_order = np.argsort(tech_importance_norm)[::-1]
ranked_descriptors = [TECH_DESCRIPTORS[i] for i in rank_order]
ranked_scores = tech_importance_norm[rank_order]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Lower Control Arm — House of Quality")
st.markdown(
    "Interactive QFD analysis for Lower Control Arm design optimisation. "
    "Adjust customer priorities in the sidebar to see real-time updates."
)
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
top_tech = TECH_DESCRIPTORS[np.argmax(tech_importance_norm)]
top_proc = PROCESSES[np.argmax(process_suitability_norm)]
total_weight = int(weight_vector.sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Top Technical Descriptor", top_tech)
col2.metric("Best Manufacturing Process", top_proc)
col3.metric("Total Customer Weight", total_weight)
col4.metric("Active Relationships", int((QFD_MATRIX > 0).sum()))

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["House of Quality", "Technical Priorities", "Process Comparison", "Data Tables"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — House of Quality Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### QFD Relationship Matrix")
    st.markdown(
        "Each cell shows the relationship strength between a customer requirement "
        "and a technical descriptor. Cell colour intensity reflects the weighted contribution."
    )

    # Weighted matrix for colour intensity
    weighted_matrix = QFD_MATRIX * weight_vector[:, np.newaxis]

    # Custom text annotations
    symbol_map = {9: "◉", 3: "○", 1: "△", 0: ""}
    annotations_text = [
        [symbol_map[int(v)] for v in row] for row in QFD_MATRIX
    ]

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=weighted_matrix,
            x=TECH_DESCRIPTORS,
            y=CUSTOMER_REQS,
            text=annotations_text,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            colorscale=[
                [0.0,  "#0e1117"],
                [0.15, "#0d2137"],
                [0.4,  "#0c4a8a"],
                [0.7,  "#1a7fd4"],
                [1.0,  "#58a6ff"],
            ],
            showscale=True,
            colorbar=dict(
                title="Weighted Score",
                tickfont=dict(color="#c9d1d9"),
            ),
            hoverongaps=False,
            hovertemplate=(
                "<b>%{y}</b> → <b>%{x}</b><br>"
                "Relationship: %{text}<br>"
                "Weighted Score: %{z:.1f}<extra></extra>"
            ),
        )
    )

    fig_heatmap.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=480,
        margin=dict(l=20, r=20, t=40, b=120),
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=12, color="#c9d1d9"),
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#c9d1d9"),
            gridcolor=GRID_COLOR,
        ),
        font=dict(color="#c9d1d9"),
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Technical importance row below heatmap
    st.markdown("#### Technical Importance Scores")
    fig_importance_row = go.Figure(
        data=go.Bar(
            x=TECH_DESCRIPTORS,
            y=tech_importance_norm,
            marker=dict(
                color=tech_importance_norm,
                colorscale=[[0, "#0c4a8a"], [1, "#58a6ff"]],
                showscale=False,
            ),
            text=[f"{v:.1f}" for v in tech_importance_norm],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=11),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
        )
    )
    fig_importance_row.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=280,
        margin=dict(l=20, r=20, t=20, b=80),
        xaxis=dict(tickangle=-35, tickfont=dict(size=11, color="#c9d1d9"), gridcolor=GRID_COLOR),
        yaxis=dict(title="Normalised Score", tickfont=dict(size=11, color="#c9d1d9"), gridcolor=GRID_COLOR),
        font=dict(color="#c9d1d9"),
    )
    st.plotly_chart(fig_importance_row, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Ranked Technical Priorities
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Ranked Technical Descriptors")
    st.markdown(
        "Descriptors are ranked by their weighted importance score — "
        "higher means more critical to meeting customer requirements."
    )

    # Colour gradient by rank
    colours = px.colors.sample_colorscale(
        "Blues", [0.3 + 0.7 * (i / (len(ranked_descriptors) - 1)) for i in range(len(ranked_descriptors))]
    )[::-1]

    fig_bar = go.Figure(
        data=go.Bar(
            x=ranked_scores,
            y=ranked_descriptors,
            orientation="h",
            marker=dict(color=colours),
            text=[f"{v:.1f}" for v in ranked_scores],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=12),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        )
    )

    fig_bar.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=460,
        margin=dict(l=20, r=80, t=20, b=40),
        xaxis=dict(
            title="Normalised Importance Score (0–100)",
            range=[0, 115],
            tickfont=dict(color="#c9d1d9"),
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(
            tickfont=dict(size=13, color="#c9d1d9"),
            gridcolor=GRID_COLOR,
        ),
        font=dict(color="#c9d1d9"),
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart
    st.markdown("### Technical Descriptor Radar")
    fig_radar = go.Figure(
        data=go.Scatterpolar(
            r=list(tech_importance_norm) + [tech_importance_norm[0]],
            theta=TECH_DESCRIPTORS + [TECH_DESCRIPTORS[0]],
            fill="toself",
            fillcolor="rgba(88, 166, 255, 0.15)",
            line=dict(color=ACCENT, width=2),
            marker=dict(color=ACCENT, size=7),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>",
        )
    )
    fig_radar.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=460,
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(
                visible=True,
                range=[0, 110],
                tickfont=dict(color="#8b949e", size=10),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(color="#c9d1d9", size=12),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
        ),
        font=dict(color="#c9d1d9"),
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Manufacturing Process Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Manufacturing Process Suitability")
    st.markdown(
        "Overall suitability is computed by weighting each process's capability "
        "scores against the current technical importance scores."
    )

    proc_colours = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]

    # Overall suitability bar
    fig_proc = go.Figure(
        data=go.Bar(
            x=PROCESSES,
            y=process_suitability_norm,
            marker=dict(color=proc_colours),
            text=[f"{v:.1f}" for v in process_suitability_norm],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=13),
            hovertemplate="<b>%{x}</b><br>Suitability: %{y:.1f}<extra></extra>",
        )
    )
    fig_proc.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        height=360,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(tickfont=dict(size=13, color="#c9d1d9"), gridcolor=GRID_COLOR),
        yaxis=dict(
            title="Normalised Suitability Score (0–100)",
            range=[0, 115],
            tickfont=dict(color="#c9d1d9"),
            gridcolor=GRID_COLOR,
        ),
        font=dict(color="#c9d1d9"),
    )
    st.plotly_chart(fig_proc, use_container_width=True)

    # Per-descriptor grouped bar
    st.markdown("### Capability per Technical Descriptor")
    fig_grouped = go.Figure()
    for i, (proc, colour) in enumerate(zip(PROCESSES, proc_colours)):
        fig_grouped.add_trace(
            go.Bar(
                name=proc,
                x=TECH_DESCRIPTORS,
                y=PROCESS_SCORES[i],
                marker_color=colour,
                hovertemplate=f"<b>{proc}</b><br>%{{x}}: %{{y}}<extra></extra>",
            )
        )

    fig_grouped.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        barmode="group",
        height=420,
        margin=dict(l=20, r=20, t=20, b=100),
        xaxis=dict(tickangle=-35, tickfont=dict(size=11, color="#c9d1d9"), gridcolor=GRID_COLOR),
        yaxis=dict(title="Capability Score (0–10)", tickfont=dict(color="#c9d1d9"), gridcolor=GRID_COLOR),
        legend=dict(
            bgcolor="#161b22",
            bordercolor=GRID_COLOR,
            font=dict(color="#c9d1d9"),
        ),
        font=dict(color="#c9d1d9"),
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    # Radar comparison
    st.markdown("### Process Capability Radar")
    fig_proc_radar = go.Figure()
    for i, (proc, colour) in enumerate(zip(PROCESSES, proc_colours)):
        r_vals = list(PROCESS_SCORES[i]) + [PROCESS_SCORES[i][0]]
        theta_vals = TECH_DESCRIPTORS + [TECH_DESCRIPTORS[0]]
        fig_proc_radar.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                fillcolor="rgba(88,166,255,0.15)",
                line=dict(color=colour, width=2),
                name=proc,
                hovertemplate=f"<b>{proc}</b><br>%{{theta}}: %{{r}}<extra></extra>",
            )
        )

    fig_proc_radar.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=500,
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(color="#8b949e", size=10),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(color="#c9d1d9", size=12),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor=GRID_COLOR,
            font=dict(color="#c9d1d9"),
        ),
        font=dict(color="#c9d1d9"),
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_proc_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Data Tables
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### QFD Relationship Matrix")
    df_qfd = pd.DataFrame(QFD_MATRIX, index=CUSTOMER_REQS, columns=TECH_DESCRIPTORS)
    st.dataframe(
        df_qfd.style.background_gradient(cmap="Blues", axis=None).format("{:.0f}"),
        use_container_width=True,
    )

    st.markdown("### Customer Weights & Technical Importance")
    df_summary = pd.DataFrame(
        {
            "Customer Requirement": CUSTOMER_REQS,
            "Priority Weight": weight_vector.astype(int),
        }
    )
    df_tech = pd.DataFrame(
        {
            "Technical Descriptor": TECH_DESCRIPTORS,
            "Importance Score": tech_importance.round(1),
            "Normalised Score": tech_importance_norm.round(1),
            "Rank": pd.Series(tech_importance_norm).rank(ascending=False).astype(int),
        }
    ).sort_values("Rank")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Customer Priorities**")
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Technical Descriptor Rankings**")
        st.dataframe(df_tech, use_container_width=True, hide_index=True)

    st.markdown("### Manufacturing Process Capability Matrix")
    df_proc = pd.DataFrame(PROCESS_SCORES, index=PROCESSES, columns=TECH_DESCRIPTORS)
    df_proc["Overall Suitability"] = process_suitability_norm.round(1)
    st.dataframe(
        df_proc.style.background_gradient(cmap="Blues", axis=None).format("{:.1f}"),
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:13px;'>"
    "Lower Control Arm QFD Optimisation Tool · Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True,
)
