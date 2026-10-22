---
title: Political Science Data Lab
emoji: 📊
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# Political Science Data Lab — Python Shiny

Interactive data science workspace for exploring political-science datasets and analyzing user-uploaded CSV files.

## Features

- Dataset registry using `data/manifest.csv`
- Interactive filters, Plotly visualizations, and filtered data preview
- Custom CSV upload with a 10 MB / 100,000-row / 150-column safety limit
- Required data dictionary validation
- Data quality profiling: missingness, unique values, duplicates, and IQR-based outlier counts
- Histogram, box plot, scatter plot, bar chart, and correlation heatmap for uploaded datasets
- Baseline regression/classification experiments with preprocessing pipelines, held-out test metrics, and leakage-aware splits
- Downloadable data-quality profile CSV
- Uploaded files are used for the current app session only and are not added to the built-in dataset registry

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m shiny run --reload --launch-browser app.py
```

## Data dictionary format

Upload a CSV with one row per dataset column. Required fields:

- `column_name`: exact column name in the dataset
- `description`: what the variable means
- `data_type`: intended type (e.g. numeric, categorical, date, text)

Optional fields: `required` and `allowed_values`. The dictionary must describe every column in the uploaded dataset. A template is available inside the app.

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space.
2. Choose **Docker** as the Space SDK.
3. Upload the contents of this project (including `Dockerfile`, `requirements.txt`, `app.py`, `data/`, and `www/`). Do not upload `.venv/` or `.git/`.
4. The Dockerfile listens on port `7860`; wait for the Space to build and open the app URL.
5. In Docusaurus, set `SHINY_APP_URL` to the resulting Space URL at build time to connect the site to this app.

## Included datasets

The bundled CSVs are marked as prototype instructional data in `data/manifest.csv`. Replace them with properly cited, documented datasets before presenting findings as real-world evidence.
