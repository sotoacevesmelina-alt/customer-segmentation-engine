import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import anthropic

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon=":dart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = (
    "You are a marketing analyst expert specializing in customer segmentation. "
    "Analyze customer segment statistics and create concise, actionable persona descriptions.\n\n"
    "For each segment provide:\n"
    "1. A memorable persona name (e.g. \"The Budget-Conscious Family Shopper\")\n"
    "2. 3-4 key behavioral traits\n"
    "3. 2-3 motivations and 2-3 pain points\n"
    "4. A short recommended marketing approach (1-2 sentences)\n\n"
    "JSON keys: persona_name (string), traits (string array), motivations (string array), "
    "pain_points (string array), marketing_approach (string)."
)

COLORS = px.colors.qualitative.Set2


# ── data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data
def cluster_data(numeric_values: np.ndarray, n_clusters: int):
    X = SimpleImputer(strategy="mean").fit_transform(numeric_values)
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)
    return labels, X_scaled


def compute_stats(df: pd.DataFrame, labels, feature_cols: list) -> list[dict]:
    tmp = df[feature_cols].copy()
    tmp["_seg"] = labels
    stats = []
    for seg_id in sorted(tmp["_seg"].unique()):
        grp = tmp[tmp["_seg"] == seg_id][feature_cols]
        stats.append({
            "segment_id": int(seg_id),
            "size": int(len(grp)),
            "percentage": round(len(grp) / len(df) * 100, 1),
            "feature_means": {c: round(float(grp[c].mean()), 2) for c in feature_cols},
        })
    return stats


_PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "persona_name": {"type": "string"},
        "traits": {"type": "array", "items": {"type": "string"}},
        "motivations": {"type": "array", "items": {"type": "string"}},
        "pain_points": {"type": "array", "items": {"type": "string"}},
        "marketing_approach": {"type": "string"},
    },
    "required": ["persona_name", "traits", "motivations", "pain_points", "marketing_approach"],
    "additionalProperties": False,
}


def generate_persona(segment: dict, all_segments: list[dict], client: anthropic.Anthropic) -> dict:
    prompt = (
        f"Segment:\n{json.dumps(segment, indent=2)}\n\n"
        "All segments (for comparison):\n"
        + json.dumps(
            [{"segment_id": s["segment_id"], "feature_means": s["feature_means"]} for s in all_segments],
            indent=2,
        )
    )
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": _PERSONA_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


# ── charts ────────────────────────────────────────────────────────────────────

def fig_scatter(X_scaled: np.ndarray, labels) -> go.Figure:
    if X_scaled.shape[1] >= 2:
        coords = PCA(n_components=2).fit_transform(X_scaled)
        x_label, y_label = "PC 1", "PC 2"
    else:
        coords = np.column_stack([X_scaled[:, 0], np.zeros(len(X_scaled))])
        x_label, y_label = "Feature Value", ""

    df_p = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Segment": [f"Segment {l}" for l in labels],
    })
    fig = px.scatter(
        df_p, x="x", y="y", color="Segment",
        color_discrete_sequence=COLORS,
        labels={"x": x_label, "y": y_label},
        title="Customer Distribution (PCA projection)",
        template="plotly_white",
        opacity=0.7,
    )
    fig.update_traces(marker_size=5)
    fig.update_layout(title_x=0, legend_title_text="Segment")
    return fig


def fig_bar(segments: list[dict]) -> go.Figure:
    df_p = pd.DataFrame({
        "Segment": [f"Segment {s['segment_id']}" for s in segments],
        "Count": [s["size"] for s in segments],
    })
    fig = px.bar(
        df_p, x="Segment", y="Count", color="Segment",
        color_discrete_sequence=COLORS,
        text="Count",
        title="Customers per Segment",
        template="plotly_white",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, title_x=0)
    return fig


def fig_heatmap(segments: list[dict], feature_cols: list[str]) -> go.Figure:
    seg_labels = [f"Seg {s['segment_id']}" for s in segments]
    matrix = np.array([[s["feature_means"][c] for c in feature_cols] for s in segments], dtype=float)
    mn, mx = matrix.min(axis=0), matrix.max(axis=0)
    norm = (matrix - mn) / np.where(mx - mn == 0, 1, mx - mn)

    fig = go.Figure(go.Heatmap(
        z=norm,
        x=feature_cols,
        y=seg_labels,
        colorscale="Blues",
        zmin=0,
        zmax=1,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Means by Segment (row-normalized)",
        template="plotly_white",
        title_x=0,
        xaxis_tickangle=-30,
    )
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Customer Segmentation")
    st.caption("K-means clustering + AI-powered personas")
    st.divider()

    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    n_clusters = st.slider("Number of Segments", min_value=2, max_value=10, value=4)

    st.divider()
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI persona generation.",
    )


# ── main ──────────────────────────────────────────────────────────────────────

st.title("Customer Segmentation")

if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    with st.expander("Expected CSV format"):
        st.markdown("""
- Any CSV with **numeric columns** (age, income, spending score, etc.)
- Non-numeric columns (IDs, names, categories) are ignored during clustering
- Missing values are imputed with column means automatically
        """)
    st.stop()

# Load + validate
file_bytes = uploaded_file.read()
try:
    df = parse_csv(file_bytes)
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.empty:
    st.error("No numeric columns found in the CSV.")
    st.stop()
if len(df) < n_clusters:
    st.error(f"Need at least {n_clusters} rows; the file has only {len(df)}.")
    st.stop()

feature_cols = numeric_df.columns.tolist()
labels, X_scaled = cluster_data(numeric_df.values, n_clusters)
segments = compute_stats(df, labels, feature_cols)


# ── overview metrics ──────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Segments", n_clusters)
c3.metric("Features Used", len(feature_cols))
st.caption(f"Clustering on: {', '.join(feature_cols)}")
st.divider()


# ── visualizations ────────────────────────────────────────────────────────────

left, right = st.columns(2)
left.plotly_chart(fig_scatter(X_scaled, labels), use_container_width=True)
right.plotly_chart(fig_bar(segments), use_container_width=True)

st.plotly_chart(fig_heatmap(segments, feature_cols), use_container_width=True)


# ── segment stats table ───────────────────────────────────────────────────────

st.subheader("Segment Statistics")
rows = [
    dict(
        {"Segment": f"Segment {s['segment_id']}", "Count": s["size"], "Share (%)": s["percentage"]},
        **{f"Avg {k}": v for k, v in s["feature_means"].items()},
    )
    for s in segments
]
st.dataframe(pd.DataFrame(rows).set_index("Segment"), use_container_width=True)
st.divider()


# ── ai personas ───────────────────────────────────────────────────────────────

st.subheader("AI-Powered Persona Descriptions")

if not api_key:
    st.warning("Enter an Anthropic API key in the sidebar to generate persona descriptions.")
else:
    cache_key = f"personas|{uploaded_file.name}|{n_clusters}|{'|'.join(feature_cols)}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = None

    if st.session_state[cache_key] is None:
        if st.button("Generate Personas", type="primary"):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                result = {}
                bar = st.progress(0, text="Generating personas...")
                for i, seg in enumerate(segments):
                    result[seg["segment_id"]] = generate_persona(seg, segments, client)
                    bar.progress((i + 1) / len(segments), text=f"Segment {i + 1} of {len(segments)} done")
                st.session_state[cache_key] = result
                st.rerun()
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Check the key entered in the sidebar.")
            except anthropic.RateLimitError:
                st.error("Rate limit reached. Wait a moment and try again.")
            except anthropic.APIStatusError as e:
                st.error(f"API error {e.status_code}: {e.message}")

    if st.session_state.get(cache_key):
        personas = st.session_state[cache_key]

        header_col, btn_col = st.columns([5, 1])
        with btn_col:
            if st.button("Regenerate", type="secondary"):
                st.session_state[cache_key] = None
                st.rerun()

        n_cols = min(n_clusters, 2)
        cols = st.columns(n_cols)
        for i, seg in enumerate(segments):
            p = personas.get(seg["segment_id"], {})
            with cols[i % n_cols]:
                with st.container(border=True):
                    st.markdown(
                        f"**Segment {seg['segment_id']}** &nbsp;·&nbsp; "
                        f"{seg['size']} customers ({seg['percentage']}%)"
                    )
                    st.markdown(f"### {p.get('persona_name', f'Segment {seg[\"segment_id\"]}')}")

                    if "traits" in p:
                        st.markdown("**Traits:** " + " · ".join(p["traits"]))

                    m_col, pp_col = st.columns(2)
                    with m_col:
                        if "motivations" in p:
                            st.markdown("**Motivations**")
                            for item in p["motivations"]:
                                st.markdown(f"- {item}")
                    with pp_col:
                        if "pain_points" in p:
                            st.markdown("**Pain Points**")
                            for item in p["pain_points"]:
                                st.markdown(f"- {item}")

                    if "marketing_approach" in p:
                        st.info(p["marketing_approach"])
