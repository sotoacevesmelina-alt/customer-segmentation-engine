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
    page_title="MarketMinds AI",
    page_icon=":dart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = (
    "You are a marketing analyst expert specializing in customer segmentation and business strategy. "
    "Analyze customer segment statistics and create detailed, actionable persona profiles tailored to "
    "the business context provided.\n\n"
    "For each segment provide:\n"
    "1. segment_name: A memorable, descriptive name (e.g. 'The Value-Driven Young Professional')\n"
    "2. demographics: age_range, income_level, and lifestyle (all strings)\n"
    "3. psychographics: motivations (2-3 items), pain_points (2-3 items), values (2-3 items)\n"
    "4. behavioral_patterns: 3-4 key behaviors derived from the segment data\n"
    "5. marketing_channels: 3-4 specific channels suited to the stated marketing budget\n"
    "6. messaging_strategy: A concise, targeted messaging approach (2-3 sentences)\n"
    "7. predicted_clv: Exactly 'High', 'Medium', or 'Low' based on segment spending and engagement\n"
    "8. campaign_recommendation: A specific, actionable campaign idea tied to the stated business goal (2-3 sentences)\n\n"
    "Tailor all recommendations to the business type, goal, and budget in the user message. "
    "Be specific and actionable, not generic."
)

COLORS = px.colors.qualitative.Set2

DATASET_OPTIONS = [
    "E-commerce Customers",
    "Small Business Customers",
    "Retail Store Customers",
    "Upload My Own CSV",
]

DEMO_NOTE = (
    "Demo datasets let you explore segmentation strategies without uploading real customer data. "
    "When you're ready, upload your own CSV to segment your actual customers."
)


# ── synthetic data generators ─────────────────────────────────────────────────

@st.cache_data
def _ecommerce_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cols = [
        "age", "annual_income", "total_spent", "purchase_frequency",
        "avg_order_value", "days_since_last_purchase", "email_engagement_score",
        "website_visits_per_month",
    ]
    # 4 clusters: young deal-seekers, affluent occasionals, mid-market regulars, senior premium
    cluster_defs = [
        (550, [24, 35000,  800, 18,  44, 15, 72, 25], [4, 8000,  300, 5,  15,  8, 15, 8]),
        (400, [47, 120000, 4200, 6,  700, 45, 45,  8], [8, 25000, 1200, 2, 200, 20, 18, 4]),
        (650, [35, 65000,  2100, 12, 175, 22, 60, 15], [6, 15000,  600, 4,  60, 10, 12, 6]),
        (400, [58, 90000,  3500,  4, 875, 60, 30,  5], [7, 20000, 1000, 2, 250, 25, 15, 3]),
    ]
    frames = []
    for n, means, stds in cluster_defs:
        data = {c: rng.normal(m, s, n) for c, m, s in zip(cols, means, stds)}
        frames.append(pd.DataFrame(data))
    df = pd.concat(frames, ignore_index=True)
    df["age"] = df["age"].clip(18, 70).round().astype(int)
    df["annual_income"] = df["annual_income"].clip(20000, 250000).round(-2).astype(int)
    df["total_spent"] = df["total_spent"].clip(50, 20000).round(2)
    df["purchase_frequency"] = df["purchase_frequency"].clip(1, 52).round().astype(int)
    df["avg_order_value"] = df["avg_order_value"].clip(10, 2000).round(2)
    df["days_since_last_purchase"] = df["days_since_last_purchase"].clip(1, 365).round().astype(int)
    df["email_engagement_score"] = df["email_engagement_score"].clip(0, 100).round(1)
    df["website_visits_per_month"] = df["website_visits_per_month"].clip(0, 100).round().astype(int)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_data
def _small_business_data() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    cols = [
        "age", "annual_income", "purchase_frequency", "satisfaction_score",
        "lifetime_value", "referral_count", "years_as_customer", "monthly_spend",
    ]
    # 4 clusters: new clients, champion loyalists, big-spend low-engagement, satisfied & thrifty
    cluster_defs = [
        (600, [32, 55000,   4, 7.0,  3000, 1, 1.2,  250], [5, 10000, 2, 1.2, 1000, 1, 0.5,  100]),
        (450, [45, 95000,  12, 9.2, 42000, 6, 8.0, 3500], [7, 20000, 3, 0.5, 8000, 2, 2.0,  800]),
        (500, [40, 130000,  8, 6.5, 28000, 2, 5.0, 4200], [6, 25000, 2, 1.0, 7000, 1, 2.0, 1000]),
        (450, [38, 60000,   7, 9.0,  8000, 4, 4.0,  500], [6, 12000, 2, 0.6, 2000, 2, 1.5,  150]),
    ]
    frames = []
    for n, means, stds in cluster_defs:
        data = {c: rng.normal(m, s, n) for c, m, s in zip(cols, means, stds)}
        frames.append(pd.DataFrame(data))
    df = pd.concat(frames, ignore_index=True)
    df["age"] = df["age"].clip(18, 70).round().astype(int)
    df["annual_income"] = df["annual_income"].clip(25000, 300000).round(-2).astype(int)
    df["purchase_frequency"] = df["purchase_frequency"].clip(1, 52).round().astype(int)
    df["satisfaction_score"] = df["satisfaction_score"].clip(1, 10).round(1)
    df["lifetime_value"] = df["lifetime_value"].clip(100, 200000).round(2)
    df["referral_count"] = df["referral_count"].clip(0, 20).round().astype(int)
    df["years_as_customer"] = df["years_as_customer"].clip(0.1, 20).round(1)
    df["monthly_spend"] = df["monthly_spend"].clip(50, 20000).round(2)
    return df.sample(frac=1, random_state=123).reset_index(drop=True)


@st.cache_data
def _retail_data() -> pd.DataFrame:
    rng = np.random.default_rng(999)
    cols = [
        "age", "visit_frequency_per_month", "avg_basket_size", "loyalty_points",
        "seasonal_shopper_score", "category_preference_score", "store_visits_per_year",
        "avg_spend_per_visit",
    ]
    # 4 clusters: frequent light shoppers, weekend bulk buyers, loyal regulars, seasonal occasionals
    cluster_defs = [
        (600, [28, 12,  25,  800, 3, 6, 140,  25], [5, 4,  8, 300, 1.5, 1.5, 30,  8]),
        (450, [42,  3, 120, 2200, 5, 7,  38, 120], [7, 1, 30, 600, 2.0, 1.5, 12, 30]),
        (550, [52,  7,  65, 5500, 4, 8,  88,  68], [8, 2, 18, 1200, 1.5, 1.2, 20, 18]),
        (400, [35,  2,  85,  400, 9, 5,  22,  90], [6, 1, 25, 200, 1.0, 1.5,  8, 25]),
    ]
    frames = []
    for n, means, stds in cluster_defs:
        data = {c: rng.normal(m, s, n) for c, m, s in zip(cols, means, stds)}
        frames.append(pd.DataFrame(data))
    df = pd.concat(frames, ignore_index=True)
    df["age"] = df["age"].clip(18, 80).round().astype(int)
    df["visit_frequency_per_month"] = df["visit_frequency_per_month"].clip(0, 30).round(1)
    df["avg_basket_size"] = df["avg_basket_size"].clip(5, 500).round(2)
    df["loyalty_points"] = df["loyalty_points"].clip(0, 20000).round().astype(int)
    df["seasonal_shopper_score"] = df["seasonal_shopper_score"].clip(1, 10).round(1)
    df["category_preference_score"] = df["category_preference_score"].clip(1, 10).round(1)
    df["store_visits_per_year"] = df["store_visits_per_year"].clip(1, 365).round().astype(int)
    df["avg_spend_per_visit"] = df["avg_spend_per_visit"].clip(5, 500).round(2)
    return df.sample(frac=1, random_state=999).reset_index(drop=True)


def load_demo_dataset(name: str) -> pd.DataFrame:
    if name == "E-commerce Customers":
        return _ecommerce_data()
    if name == "Small Business Customers":
        return _small_business_data()
    return _retail_data()


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


# ── ai personas ───────────────────────────────────────────────────────────────

_PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "segment_name": {"type": "string"},
        "demographics": {
            "type": "object",
            "properties": {
                "age_range": {"type": "string"},
                "income_level": {"type": "string"},
                "lifestyle": {"type": "string"},
            },
            "required": ["age_range", "income_level", "lifestyle"],
            "additionalProperties": False,
        },
        "psychographics": {
            "type": "object",
            "properties": {
                "motivations": {"type": "array", "items": {"type": "string"}},
                "pain_points": {"type": "array", "items": {"type": "string"}},
                "values": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["motivations", "pain_points", "values"],
            "additionalProperties": False,
        },
        "behavioral_patterns": {"type": "array", "items": {"type": "string"}},
        "marketing_channels": {"type": "array", "items": {"type": "string"}},
        "messaging_strategy": {"type": "string"},
        "predicted_clv": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "campaign_recommendation": {"type": "string"},
    },
    "required": [
        "segment_name", "demographics", "psychographics", "behavioral_patterns",
        "marketing_channels", "messaging_strategy", "predicted_clv", "campaign_recommendation",
    ],
    "additionalProperties": False,
}


def generate_persona(
    segment: dict,
    all_segments: list[dict],
    client: anthropic.Anthropic,
    business_ctx: dict,
) -> dict:
    active_ctx = {k: v for k, v in business_ctx.items() if v}
    biz_block = ("\n\nBusiness Context:\n" + json.dumps(active_ctx, indent=2)) if active_ctx else ""
    prompt = (
        f"Segment:\n{json.dumps(segment, indent=2)}\n\n"
        "All segments (for comparison):\n"
        + json.dumps(
            [{"segment_id": s["segment_id"], "feature_means": s["feature_means"]} for s in all_segments],
            indent=2,
        )
        + biz_block
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
    st.title("MarketMinds AI")
    st.caption("AI-powered customer segmentation with intelligent personas")
    st.divider()

    with st.expander("Tell us about your business", expanded=False):
        biz_type = st.text_input(
            "Business type",
            placeholder="e.g., Local coffee shop, Online boutique, SaaS startup",
        )
        age_col1, age_col2 = st.columns(2)
        with age_col1:
            age_min = st.number_input("Min age", min_value=13, max_value=100, value=18, step=1)
        with age_col2:
            age_max = st.number_input("Max age", min_value=13, max_value=100, value=65, step=1)
        biz_goal = st.selectbox(
            "Primary business goal",
            [
                "",
                "Increase sales",
                "Retain existing customers",
                "Expand to new markets",
                "Launch new product",
                "Improve brand awareness",
            ],
        )
        biz_budget = st.selectbox(
            "Current marketing budget",
            ["", "Under $1K/month", "$1K-$5K/month", "$5K-$20K/month", "$20K+/month"],
        )

    has_biz_context = bool(biz_type or biz_goal or biz_budget)
    business_ctx = {
        "business_type": biz_type or None,
        "target_age_range": f"{age_min}–{age_max}" if has_biz_context else None,
        "business_goal": biz_goal or None,
        "marketing_budget": biz_budget or None,
    }

    st.markdown("**Data Source**")
    dataset_choice = st.selectbox("Dataset", DATASET_OPTIONS, index=0, label_visibility="collapsed")
    if dataset_choice != "Upload My Own CSV":
        st.caption(DEMO_NOTE)

    uploaded_file = None
    if dataset_choice == "Upload My Own CSV":
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

st.title("MarketMinds AI")

# ── data loading ──────────────────────────────────────────────────────────────

if dataset_choice == "Upload My Own CSV":
    if uploaded_file is None:
        st.info("Upload a CSV file in the sidebar to get started.")
        with st.expander("Expected CSV format"):
            st.markdown("""
- Any CSV with **numeric columns** (age, income, spending score, etc.)
- Non-numeric columns (IDs, names, categories) are ignored during clustering
- Missing values are imputed with column means automatically
            """)
        st.stop()
    file_bytes = uploaded_file.read()
    try:
        df = parse_csv(file_bytes)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()
    data_id = uploaded_file.name
else:
    df = load_demo_dataset(dataset_choice)
    data_id = dataset_choice

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
st.caption(f"Dataset: {dataset_choice} · Clustering on: {', '.join(feature_cols)}")
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

CLV_BADGE = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}

if not api_key:
    st.warning("Enter an Anthropic API key in the sidebar to generate persona descriptions.")
else:
    biz_str = json.dumps({k: v for k, v in business_ctx.items() if v}, sort_keys=True)
    cache_key = f"personas|{data_id}|{n_clusters}|{biz_str}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = None

    if st.session_state[cache_key] is None:
        if st.button("Generate Personas", type="primary"):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                result = {}
                bar = st.progress(0, text="Generating personas...")
                for i, seg in enumerate(segments):
                    result[seg["segment_id"]] = generate_persona(seg, segments, client, business_ctx)
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
                    st.markdown(f"### {p.get('segment_name', 'Segment ' + str(seg['segment_id']))}")

                    clv = p.get("predicted_clv", "")
                    if clv:
                        st.markdown(f"**Predicted CLV:** {CLV_BADGE.get(clv, clv)}")

                    demo = p.get("demographics", {})
                    if demo:
                        st.markdown(
                            f"**Demographics:** {demo.get('age_range', '')} · "
                            f"{demo.get('income_level', '')} · {demo.get('lifestyle', '')}"
                        )

                    psych = p.get("psychographics", {})
                    if psych:
                        m_col, pp_col = st.columns(2)
                        with m_col:
                            if psych.get("motivations"):
                                st.markdown("**Motivations**")
                                for item in psych["motivations"]:
                                    st.markdown(f"- {item}")
                        with pp_col:
                            if psych.get("pain_points"):
                                st.markdown("**Pain Points**")
                                for item in psych["pain_points"]:
                                    st.markdown(f"- {item}")
                        if psych.get("values"):
                            st.markdown("**Values:** " + " · ".join(psych["values"]))

                    if p.get("behavioral_patterns"):
                        st.markdown("**Behavioral Patterns**")
                        for item in p["behavioral_patterns"]:
                            st.markdown(f"- {item}")

                    if p.get("marketing_channels"):
                        st.markdown("**Marketing Channels:** " + " · ".join(p["marketing_channels"]))

                    if p.get("messaging_strategy"):
                        st.info(p["messaging_strategy"])

                    if p.get("campaign_recommendation"):
                        st.success(p["campaign_recommendation"])
