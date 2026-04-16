# Customer Segmentation

Streamlit app that clusters customers from a CSV upload using K-means and generates AI-powered persona descriptions via the Anthropic API.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Upload a CSV with customer data (any numeric columns work — age, income, spending score, etc.)
2. Use the sidebar slider to choose the number of segments (2–10)
3. The app immediately shows:
   - PCA scatter plot of customer distribution
   - Segment size bar chart
   - Feature means heatmap across segments
   - Full segment statistics table
4. Enter your Anthropic API key and click **Generate Personas** to get AI-written descriptions for each segment

## CSV requirements

| Requirement | Detail |
|---|---|
| Format | `.csv` |
| Numeric columns | At least one required — used as clustering features |
| Non-numeric columns | Ignored automatically (IDs, names, categories) |
| Missing values | Imputed with column means |

## How it works

1. **Parse** — CSV is loaded with pandas; numeric columns are selected automatically.
2. **Cluster** — Columns are scaled with `StandardScaler` and fed to `KMeans`. Results are cached so changing the sidebar slider is instant.
3. **Visualize** — Three Plotly charts show distribution (PCA), segment sizes, and normalized feature means.
4. **Persona** — For each cluster, `claude-sonnet-4-6` generates a structured persona. The system prompt uses `cache_control: ephemeral` to reduce latency and cost across per-segment calls.
