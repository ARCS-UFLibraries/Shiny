from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from analytics import profile_dataframe, validate_data_dictionary

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"


def load_manifest():
    return pd.read_csv(MANIFEST_PATH)


def load_dataset(dataset_id):
    m = load_manifest()
    row = m.loc[m.dataset_id == dataset_id]
    if row.empty:
        raise ValueError(f"Dataset '{dataset_id}' is not in the data registry.")
    path = DATA_DIR / str(row.iloc[0]["file"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    return pd.read_csv(path)


def dataset_info(dataset_id):
    df_m = load_manifest()
    match = df_m.loc[df_m.dataset_id == dataset_id]
    if match.empty:
        return df_m.iloc[0]
    return match.iloc[0]


def get_column_types(df):
    """Categorize columns into discrete/categorical factors and continuous/numeric metrics."""
    discrete = []
    numeric = []
    for col in df.columns:
        if df[col].dtype == "object" or (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10):
            discrete.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
    # If all numeric columns have <= 10 unique values, allow them as metrics as well
    if not numeric:
        numeric = df.select_dtypes(include="number").columns.tolist()
    return discrete, numeric


def build_filter_inputs(df, prefix="filter_"):
    controls = []
    discrete_cols, numeric_cols = get_column_types(df)
    
    # 1. Discrete / Category Multi-select dropdowns
    for col in discrete_cols[:6]:
        vals = sorted(df[col].dropna().astype(str).unique())
        if 1 < len(vals) <= 50:
            clean_name = col.replace("_", " ").title()
            controls.append(
                ui.div(
                    ui.tags.label(f"{clean_name}", class_="sidebar-label"),
                    ui.input_selectize(
                        f"{prefix}{col}",
                        None,
                        choices=vals,
                        multiple=True,
                        options={"placeholder": f"All {clean_name.lower()}s included..."},
                    ),
                    class_="filter-item",
                )
            )

    # 2. Continuous Numeric Sliders
    for col in numeric_cols[:3]:
        valid = df[col].dropna()
        if not valid.empty:
            lo, hi = float(valid.min()), float(valid.max())
            if lo != hi:
                clean_name = col.replace("_", " ").title()
                step_val = 1 if (hi - lo) > 10 else 0.5
                controls.append(
                    ui.div(
                        ui.tags.label(f"{clean_name} Range", class_="sidebar-label"),
                        ui.input_slider(
                            f"{prefix}range_{col}",
                            None,
                            min=lo,
                            max=hi,
                            value=(lo, hi),
                            step=step_val,
                        ),
                        class_="filter-item",
                    )
                )
    return controls or [ui.p("No filters are required for this dataset.", class_="text-muted small")]


CSS = """
:root {
  --uf-blue: #0021A5;
  --uf-blue-dark: #001773;
  --uf-blue-soft: #edf2ff;
  --uf-orange: #FA4616;
  --uf-orange-dark: #d83407;
  --uf-orange-soft: #fff5f2;
  --ink-primary: #0a101d;
  --ink-secondary: #475467;
  --ink-muted: #667085;
  --canvas-bg: #f8fafc;
  --card-bg: #ffffff;
  --border-subtle: #e4e7ec;
  --border-focus: #0021A5;
  --radius-xl: 18px;
  --radius-lg: 12px;
  --radius-md: 8px;
  --radius-full: 9999px;
  --shadow-sm: 0 1px 3px rgba(16, 24, 40, 0.05);
  --shadow-card: 0 4px 20px -2px rgba(0, 33, 165, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
  --shadow-float: 0 12px 30px -4px rgba(0, 33, 165, 0.12);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background-color: var(--canvas-bg);
  color: var(--ink-primary);
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
  letter-spacing: -0.01em;
  -webkit-font-smoothing: antialiased;
}

/* UF Top Navbar */
.uf-navbar {
  background: var(--uf-blue);
  color: #ffffff;
  padding: 16px 36px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 2px 14px rgba(0, 33, 165, 0.25);
  position: sticky;
  top: 0;
  z-index: 1000;
  border-bottom: 3px solid var(--uf-orange);
}
.uf-brand {
  display: flex;
  align-items: center;
  gap: 16px;
}
.uf-logo-badge {
  background: #ffffff;
  color: var(--uf-blue);
  font-weight: 900;
  font-size: 17px;
  padding: 7px 14px;
  border-radius: var(--radius-md);
  letter-spacing: 0.04em;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}
.uf-brand-title {
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0;
  color: #ffffff;
}
.uf-brand-sub {
  font-size: 12px;
  color: #dbeafe;
  margin: 0;
  font-weight: 500;
}
.uf-nav-badge {
  background: rgba(255, 255, 255, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.25);
  color: #ffffff;
  font-size: 11px;
  font-weight: 700;
  padding: 6px 14px;
  border-radius: var(--radius-full);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* Page Frame */
.uf-container {
  max-width: 1440px;
  margin: 26px auto 60px;
  padding: 0 28px;
}

/* UF Hero Showcase Banner */
.uf-hero {
  background: linear-gradient(135deg, var(--uf-blue) 0%, #001773 100%);
  border-radius: var(--radius-xl);
  padding: 34px 40px;
  color: #ffffff;
  box-shadow: var(--shadow-float);
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 20px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.uf-hero::after {
  content: "";
  position: absolute;
  right: -5%;
  bottom: -40%;
  width: 340px;
  height: 340px;
  background: radial-gradient(circle, rgba(250, 70, 22, 0.25) 0%, transparent 70%);
  pointer-events: none;
}
.uf-hero-tag {
  display: inline-block;
  background: var(--uf-orange);
  color: #ffffff;
  font-size: 11px;
  font-weight: 800;
  padding: 4px 12px;
  border-radius: var(--radius-full);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.uf-hero h1 {
  font-size: 26px;
  font-weight: 800;
  margin: 0 0 8px 0;
  color: #ffffff;
  letter-spacing: -0.02em;
}
.uf-hero p {
  color: #cbd5e1;
  font-size: 14px;
  max-width: 680px;
  margin: 0;
  line-height: 1.6;
}
.uf-workflow-steps {
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(8px);
  padding: 10px 18px;
  border-radius: var(--radius-full);
  border: 1px solid rgba(255, 255, 255, 0.2);
}
.uf-step-num {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--uf-orange);
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 800;
}
.uf-step-label {
  font-size: 13px;
  font-weight: 600;
  color: #ffffff;
}
.uf-step-arrow {
  color: rgba(255, 255, 255, 0.5);
  font-size: 14px;
}

/* Tabs */
.nav-tabs {
  border: none;
  background: #ffffff;
  padding: 6px;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-subtle);
  gap: 6px;
  margin-bottom: 24px;
  display: inline-flex;
}
.nav-tabs .nav-link {
  color: var(--ink-secondary);
  font-weight: 600;
  font-size: 14px;
  padding: 10px 22px;
  border: none;
  border-radius: var(--radius-md);
  transition: all 0.15s ease;
}
.nav-tabs .nav-link:hover {
  color: var(--uf-blue);
  background: var(--uf-blue-soft);
}
.nav-tabs .nav-link.active {
  background: var(--uf-blue);
  color: #ffffff;
  font-weight: 700;
  box-shadow: 0 2px 8px rgba(0, 33, 165, 0.2);
}

/* KPI Summary Deck */
.uf-kpi-deck {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.uf-kpi-card {
  background: #ffffff;
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  border: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-sm);
  border-left: 4px solid var(--uf-blue);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.uf-kpi-card.orange {
  border-left-color: var(--uf-orange);
}
.uf-kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-card);
}
.uf-kpi-val {
  font-size: 24px;
  font-weight: 800;
  color: var(--uf-blue);
  line-height: 1.1;
  letter-spacing: -0.02em;
}
.uf-kpi-card.orange .uf-kpi-val {
  color: var(--uf-orange-dark);
}
.uf-kpi-lbl {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-muted);
  margin-top: 6px;
}

/* Card Panels */
.uf-card {
  background: #ffffff;
  border-radius: var(--radius-xl);
  border: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-card);
  padding: 26px 30px;
  margin-bottom: 24px;
}
.uf-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-subtle);
}
.uf-card-title {
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--uf-blue);
  margin: 0 0 3px 0;
}
.uf-card-sub {
  font-size: 13px;
  color: var(--ink-secondary);
  margin: 0;
}

/* Controls Toolbar */
.uf-toolbar {
  background: var(--uf-blue-soft);
  border: 1px solid #dbeafe;
  border-radius: var(--radius-lg);
  padding: 16px 20px;
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-end;
  margin-bottom: 20px;
}
.uf-toolbar .form-group {
  margin-bottom: 0;
  flex: 1;
  min-width: 170px;
}
.uf-toolbar label {
  font-size: 12px;
  font-weight: 700;
  color: var(--uf-blue-dark);
  margin-bottom: 5px;
  display: block;
}

/* Sidebar Customization */
.bslib-sidebar-panel {
  background: #ffffff !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-xl) !important;
  padding: 22px !important;
  box-shadow: var(--shadow-sm) !important;
}
.sidebar-badge {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--uf-blue);
  background: var(--uf-blue-soft);
  padding: 4px 10px;
  border-radius: var(--radius-md);
  display: inline-block;
  margin-bottom: 12px;
}
.sidebar-label {
  font-size: 12px;
  font-weight: 700;
  color: var(--ink-secondary);
  margin-bottom: 4px;
  display: block;
}
.filter-item {
  margin-bottom: 14px;
}

/* Analytical Insight Note */
.uf-insight-box {
  background: var(--uf-orange-soft);
  border: 1px solid #fed7aa;
  border-left: 4px solid var(--uf-orange);
  border-radius: var(--radius-lg);
  padding: 18px 22px;
  margin-top: 20px;
}
.uf-insight-title {
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--uf-orange-dark);
  margin-bottom: 4px;
}
.uf-insight-box p {
  margin: 0;
  font-size: 13.5px;
  color: #7c2d12;
  line-height: 1.5;
}

/* UF Buttons */
.btn-uf-primary {
  background: var(--uf-blue) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  padding: 10px 20px !important;
  box-shadow: 0 2px 6px rgba(0, 33, 165, 0.25) !important;
  transition: all 0.15s ease !important;
}
.btn-uf-primary:hover {
  background: var(--uf-blue-dark) !important;
  transform: translateY(-1px);
}
.btn-uf-orange {
  background: var(--uf-orange) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  padding: 10px 20px !important;
  box-shadow: 0 2px 6px rgba(250, 70, 22, 0.25) !important;
}
.btn-uf-orange:hover {
  background: var(--uf-orange-dark) !important;
}
.btn-uf-outline {
  background: #ffffff !important;
  color: var(--uf-blue) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  padding: 8px 16px !important;
  transition: all 0.15s ease !important;
}
.btn-uf-outline:hover {
  background: var(--uf-blue-soft) !important;
  border-color: #bfdbfe !important;
}

/* Summary Grid */
.uf-summary-list {
  background: #f8fafc;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 16px 20px;
  font-size: 13px;
}
.uf-summary-row {
  display: flex;
  justify-content: space-between;
  padding: 5px 0;
  border-bottom: 1px solid #f1f5f9;
}
.uf-summary-row:last-child {
  border-bottom: none;
}
.uf-summary-lbl {
  color: var(--ink-secondary);
  font-weight: 600;
}
.uf-summary-val {
  font-weight: 700;
  color: var(--uf-blue);
}

/* Footer */
.uf-footer {
  text-align: center;
  padding: 32px 16px;
  color: var(--ink-muted);
  font-size: 12.5px;
  border-top: 1px solid var(--border-subtle);
  margin-top: 48px;
}
"""

# Official University of Florida Visual Palette
UF_PALETTE = ["#0021A5", "#FA4616", "#007A33", "#5C768D", "#D35400", "#1E3A8A"]

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.title("UF Social Science Data Lab"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin="anonymous"),
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap",
        ),
        ui.tags.style(CSS),
    ),
    # UF Branded Top Navigation Bar
    ui.div(
        ui.div(
            ui.span("UF", class_="uf-logo-badge"),
            ui.div(
                ui.p("University of Florida • Social Science Data Lab", class_="uf-brand-title"),
                ui.p("Interactive Empirical Visualizations & Research Platform", class_="uf-brand-sub"),
            ),
            class_="uf-brand",
        ),
        ui.span("RESEARCH & LEARNING SUITE", class_="uf-nav-badge"),
        class_="uf-navbar",
    ),
    ui.div(
        ui.navset_tab(
            # TAB 1: CURATED SOCIAL SCIENCE TOPICS
            ui.nav_panel(
                "Explore Curated Datasets",
                ui.div(
                    ui.div(
                        ui.div(
                            ui.span("UF Data Science Initiative", class_="uf-hero-tag"),
                            ui.h1("Empirical Social Science Data Explorer"),
                            ui.p(
                                "Investigate voter turnout differences, regional public opinion dynamics, and community participation patterns across the United States."
                            ),
                        ),
                        ui.div(
                            ui.div(
                                ui.div("1", class_="uf-step-num"),
                                ui.span("Pick Topic", class_="uf-step-label"),
                            ),
                            ui.span("→", class_="uf-step-arrow"),
                            ui.div(
                                ui.div("2", class_="uf-step-num"),
                                ui.span("Adjust Filters", class_="uf-step-label"),
                            ),
                            ui.span("→", class_="uf-step-arrow"),
                            ui.div(
                                ui.div("3", class_="uf-step-num"),
                                ui.span("Examine Trends", class_="uf-step-label"),
                            ),
                            class_="uf-workflow-steps",
                        ),
                        class_="uf-hero",
                    ),
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div("Topic Registry", class_="sidebar-badge"),
                            ui.input_select("dataset", "Dataset to Explore", choices={}),
                            ui.output_ui("dataset_description"),
                            ui.hr(style="margin: 16px 0; border-color: var(--border-subtle);"),
                            ui.div("Dataset Filters", class_="sidebar-badge"),
                            ui.output_ui("dynamic_filters"),
                            ui.hr(style="margin: 16px 0; border-color: var(--border-subtle);"),
                            ui.input_action_button("reset", "↺ Reset All Filters", class_="btn-uf-outline", width="100%"),
                            open="open",
                            width="310px",
                        ),
                        # KPI Deck
                        ui.div(
                            ui.div(
                                ui.div(ui.output_text("dataset_kpi"), class_="uf-kpi-val", style="font-size: 18px;"),
                                ui.div("Selected Topic", class_="uf-kpi-lbl"),
                                class_="uf-kpi-card",
                            ),
                            ui.div(
                                ui.div(ui.output_text("row_kpi"), class_="uf-kpi-val"),
                                ui.div("Active Records", class_="uf-kpi-lbl"),
                                class_="uf-kpi-card",
                            ),
                            ui.div(
                                ui.div(ui.output_text("quick_stat_kpi"), class_="uf-kpi-val"),
                                ui.div("Primary Measure Average", class_="uf-kpi-lbl"),
                                class_="uf-kpi-card orange",
                            ),
                            class_="uf-kpi-deck",
                        ),
                        # Main Visualization Card
                        ui.div(
                            ui.div(
                                ui.div(
                                    ui.h2("Interactive Visualization Canvas", class_="uf-card-title"),
                                    ui.p("Compare groups and analyze distributions across social dimensions.", class_="uf-card-sub"),
                                ),
                                ui.input_select(
                                    "chart_type",
                                    None,
                                    choices={
                                        "bar": "Bar Chart (Group Comparisons)",
                                        "line": "Line Chart (Trend / Sequence)",
                                        "scatter": "Scatter Plot (Bivariate Correlation)",
                                        "hist": "Distribution (Histogram)",
                                        "box": "Box Plot (Quartiles & Outliers)",
                                    },
                                    selected="bar",
                                ),
                                class_="uf-card-header",
                            ),
                            ui.output_ui("chart_controls"),
                            ui.div(output_widget("plot"), style="min-height: 440px;"),
                            ui.div(
                                ui.div("UF Sociological Reflection & Inquiry", class_="uf-insight-title"),
                                ui.output_ui("pedagogical_prompt"),
                                class_="uf-insight-box",
                            ),
                            class_="uf-card",
                        ),
                        # Data Records Table
                        ui.div(
                            ui.h2("Observation Records Table", class_="uf-card-title"),
                            ui.p("Detailed view of records synchronized in real-time with current filters.", class_="uf-card-sub"),
                            ui.output_data_frame("table"),
                            class_="uf-card",
                        ),
                    ),
                ),
            ),
            # TAB 2: UPLOAD CUSTOM DATASET
            ui.nav_panel(
                "Upload Custom Dataset",
                ui.div(
                    ui.div(
                        ui.div(
                            ui.span("UF Research Workspace", class_="uf-hero-tag"),
                            ui.h1("Custom Dataset Analysis & Quality Audit"),
                            ui.p(
                                "Upload any CSV spreadsheet to automatically perform schema validation, apply multi-column dynamic filters, and generate multi-variable visual charts."
                            ),
                        ),
                        class_="uf-hero",
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("1. Upload Files"),
                            ui.input_file("upload_csv", "Select CSV File (Max 10 MB)", accept=[".csv"], multiple=False),
                            ui.input_file("upload_dictionary", "Optional Data Dictionary (CSV)", accept=[".csv"], multiple=False),
                            ui.tags.p(
                                "A data dictionary outlines variable definitions and allowed values. Download the template below if desired.",
                                class_="text-muted small",
                            ),
                            ui.tags.div(
                                ui.tags.a(
                                    "Download Dictionary Template (CSV)",
                                    href="/data_dictionary_template.csv",
                                    download="data_dictionary_template.csv",
                                    class_="btn-uf-outline",
                                    style="display: inline-block; text-decoration: none; margin-bottom: 12px; font-size: 12px;",
                                ),
                            ),
                            ui.output_ui("upload_status"),
                            ui.download_button("download_quality_report", "Download Quality Audit CSV", class_="btn-uf-outline"),
                        ),
                        ui.card(
                            ui.card_header("2. Dataset Overview"),
                            ui.output_ui("upload_quality_summary"),
                        ),
                        col_widths=(6, 6),
                    ),
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div("Uploaded Data Filters", class_="sidebar-badge"),
                            ui.output_ui("upload_dynamic_filters"),
                            ui.hr(style="margin: 16px 0; border-color: var(--border-subtle);"),
                            ui.input_action_button("upload_reset", "↺ Reset Upload Filters", class_="btn-uf-outline", width="100%"),
                            open="open",
                            width="310px",
                        ),
                        ui.div(
                            ui.div(
                                ui.div(
                                    ui.h2("Custom Dataset Visualization", class_="uf-card-title"),
                                    ui.p("Select presentation style and configure dimensions.", class_="uf-card-sub"),
                                ),
                                ui.input_select(
                                    "upload_chart_type",
                                    None,
                                    choices={
                                        "bar": "Bar Chart (Compare Categories)",
                                        "line": "Line Chart (Trend / Sequence)",
                                        "scatter": "Scatter Plot (Correlation with Trendline)",
                                        "hist": "Distribution (Histogram)",
                                        "box": "Box Plot (Spread & Outliers)",
                                        "heatmap": "Correlation Heatmap Matrix",
                                    },
                                    selected="bar",
                                ),
                                class_="uf-card-header",
                            ),
                            ui.output_ui("upload_chart_controls"),
                            output_widget("upload_plot"),
                            class_="uf-card",
                        ),
                        ui.div(
                            ui.h2("Filtered Observations", class_="uf-card-title"),
                            ui.p("Spreadsheet rows updated dynamically as you filter above.", class_="uf-card-sub"),
                            ui.output_data_frame("upload_preview"),
                            class_="uf-card",
                        ),
                    ),
                    ui.div(
                        ui.h2("Variable Health & Schema Audit", class_="uf-card-title"),
                        ui.p("Displays missing counts, unique values, and statistical outliers (IQR) per column.", class_="uf-card-sub"),
                        ui.output_data_frame("upload_quality_table"),
                        class_="uf-card",
                    ),
                ),
            ),
            # TAB 3: PREDICTIVE MODELING
            ui.nav_panel(
                "Predictive Modeling",
                ui.div(
                    ui.div(
                        ui.div(
                            ui.span("Baseline Intelligence", class_="uf-hero-tag"),
                            ui.h1("Predictive Modeling Pipeline"),
                            ui.p(
                                "Train transparent Random Forest models on your uploaded data using held-out test splits and preprocessing pipelines."
                            ),
                        ),
                        class_="uf-hero",
                    ),
                    ui.div(
                        ui.h2("Model Setup & Evaluation", class_="uf-card-title"),
                        ui.p("Upload a dataset in the previous tab, choose the response target column, and fit the model.", class_="uf-card-sub"),
                        ui.layout_columns(
                            ui.input_select("ml_target", "Target Column (Response Variable)", choices={}),
                            ui.input_select(
                                "ml_task",
                                "Task Type",
                                choices={"classification": "Classification (Categorical Target)", "regression": "Regression (Continuous Target)"},
                                selected="classification",
                            ),
                            ui.div(
                                ui.tags.label("Execute", class_="sidebar-label"),
                                ui.input_action_button("run_ml", "Train Baseline Model", class_="btn-uf-primary", width="100%"),
                            ),
                            col_widths=(5, 4, 3),
                        ),
                        ui.div(ui.output_ui("ml_status"), style="margin-top: 16px;"),
                        ui.div(ui.output_data_frame("ml_metrics"), style="margin-top: 16px;"),
                        class_="uf-card",
                    ),
                ),
            ),
            # TAB 4: REGISTRY & METHODOLOGY
            ui.nav_panel(
                "Registry & Methodology",
                ui.div(
                    ui.div(
                        ui.div(
                            ui.span("Academic Standards", class_="uf-hero-tag"),
                            ui.h1("Data Architecture & Replication"),
                            ui.p(
                                "Transparent registry manifest documentation, data schemas, and replication instructions."
                            ),
                        ),
                        class_="uf-hero",
                    ),
                    ui.div(
                        ui.h2("Curated Data Registry", class_="uf-card-title"),
                        ui.p("All datasets registered in data/manifest.csv.", class_="uf-card-sub"),
                        ui.output_data_frame("manifest_table"),
                        class_="uf-card",
                    ),
                    ui.div(
                        ui.h2("Adding New Datasets", class_="uf-card-title"),
                        ui.tags.ol(
                            ui.tags.li(
                                ui.tags.strong("Add CSV to data/ directory: "),
                                "Ensure standard formatting and clean headers.",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Register in data/manifest.csv: "),
                                "Include dataset_id, dataset_name, file, description, category, and source.",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Automatic Discovery: "),
                                "The application automatically detects variables, factor types, and filter ranges.",
                            ),
                            style="line-height: 1.8; font-size: 14px; color: var(--ink-secondary);",
                        ),
                        class_="uf-card",
                    ),
                ),
            ),
        ),
        class_="uf-container",
    ),
    ui.div(
        ui.p("University of Florida • Social Science Data Lab • Ready for Localhost & Cloud Deployment on Render"),
        class_="uf-footer",
    ),
)


def server(input, output, session):
    # Manifest & Curated Datasets
    @reactive.calc
    def manifest():
        return load_manifest()

    @reactive.effect
    def _update_choices():
        m = manifest()
        choices = dict(zip(m.dataset_id, m.dataset_name))
        ui.update_select("dataset", choices=choices, selected=m.iloc[0].dataset_id)

    @reactive.calc
    def raw_data():
        return load_dataset(input.dataset())

    @reactive.effect
    @reactive.event(input.reset)
    def _reset_filters():
        df = raw_data()
        discrete_cols, numeric_cols = get_column_types(df)
        for col in discrete_cols:
            try:
                ui.update_selectize(f"filter_{col}", selected=[])
            except Exception:
                pass
        for col in numeric_cols:
            try:
                valid = df[col].dropna()
                if not valid.empty:
                    lo, hi = float(valid.min()), float(valid.max())
                    if lo != hi:
                        ui.update_slider(f"filter_range_{col}", value=(lo, hi))
            except Exception:
                pass

    @reactive.calc
    def filtered_data():
        df = raw_data().copy()
        discrete_cols, numeric_cols = get_column_types(df)
        
        # Apply categorical/discrete filters
        for col in discrete_cols:
            try:
                selected = input[f"filter_{col}"]()
            except Exception:
                selected = None
            if selected:
                df = df[df[col].astype(str).isin(selected)]
                
        # Apply numeric range sliders
        for col in numeric_cols:
            try:
                value = input[f"filter_range_{col}"]()
            except Exception:
                value = None
            if value is not None and len(value) == 2:
                df = df[df[col].between(value[0], value[1])]
                
        return df

    @output
    @render.ui
    def dataset_description():
        r = dataset_info(input.dataset())
        return ui.div(
            ui.p(r.description, class_="text-muted small", style="margin-top: 8px; line-height: 1.5;"),
            ui.p(ui.strong("Category: "), str(r.get("category", "Social Science")), class_="text-muted small"),
            ui.p(ui.strong("Source: "), str(r.source), class_="text-muted small"),
        )

    @output
    @render.ui
    def dynamic_filters():
        return ui.div(*build_filter_inputs(raw_data(), prefix="filter_"))

    @output
    @render.text
    def row_kpi():
        return f"{len(filtered_data()):,} records"

    @output
    @render.text
    def dataset_kpi():
        return str(dataset_info(input.dataset()).dataset_name)

    @output
    @render.text
    def quick_stat_kpi():
        df = filtered_data()
        nums = df.select_dtypes(include="number").columns.tolist()
        if not nums or df.empty:
            return "N/A"
        col = nums[0]
        avg = df[col].mean()
        label = col.replace("_", " ").title()
        return f"{avg:.1f}% ({label})" if "percent" in col or "rate" in col else f"{avg:.2f} ({label})"

    @output
    @render.ui
    def pedagogical_prompt():
        ds = input.dataset()
        if ds == "elections":
            return ui.p(
                "Observe how participation rates vary across election years and states. Notice how differences in mail ballot availability correlate with overall voter turnout."
            )
        elif ds == "public_opinion":
            return ui.p(
                "Examine the regional distribution across policy attitudes. Consider what cultural, demographic, and economic factors shape public opinion in different geographic regions."
            )
        elif ds == "participation":
            return ui.p(
                "Notice the gradient in civic involvement across education levels and age brackets. How does one's community environment (urban vs. rural) create different avenues for community participation?"
            )
        return ui.p(
            "Analyze the primary correlation or group differences. What underlying social mechanism could explain the observed variance?"
        )

    @output
    @render.ui
    def chart_controls():
        df = filtered_data()
        discrete_cols, numeric_cols = get_column_types(df)
        chart = input.chart_type()

        num_choices = {c: c.replace("_", " ").title() for c in numeric_cols}
        disc_choices = {c: c.replace("_", " ").title() for c in discrete_cols}

        if chart == "scatter":
            x_default = numeric_cols[0] if numeric_cols else df.columns[0]
            y_default = numeric_cols[1] if len(numeric_cols) > 1 else x_default
            return ui.div(
                ui.input_select("x_col", "X-Axis Variable", choices=num_choices or disc_choices, selected=x_default),
                ui.input_select("y_col", "Y-Axis Variable", choices=num_choices or disc_choices, selected=y_default),
                ui.input_select("color_col", "Color Grouping", choices={"(None)": "None (UF Blue)"} | disc_choices, selected="(None)"),
                class_="uf-toolbar",
            )
        if chart in ("bar", "line"):
            x_default = discrete_cols[0] if discrete_cols else df.columns[0]
            y_default = numeric_cols[0] if numeric_cols else df.columns[-1]
            sub_cats = {"(None)": "None"} | {c: disc_choices[c] for c in discrete_cols if c != x_default}
            return ui.div(
                ui.input_select("x_col", "Category Dimension", choices=disc_choices or num_choices, selected=x_default),
                ui.input_select("y_col", "Numeric Measure (Mean)", choices=num_choices or disc_choices, selected=y_default),
                ui.input_select("color_col", "Sub-Group (Color)", choices=sub_cats, selected="(None)"),
                class_="uf-toolbar",
            )
        if chart == "hist":
            x_default = numeric_cols[0] if numeric_cols else df.columns[0]
            return ui.div(
                ui.input_select("x_col", "Distribution Variable", choices=num_choices or disc_choices, selected=x_default),
                ui.input_select("color_col", "Stratify by Group", choices={"(None)": "None"} | disc_choices, selected="(None)"),
                class_="uf-toolbar",
            )
        if chart == "box":
            y_default = numeric_cols[0] if numeric_cols else df.columns[0]
            x_default = discrete_cols[0] if discrete_cols else "(None)"
            return ui.div(
                ui.input_select("y_col", "Continuous Variable", choices=num_choices or disc_choices, selected=y_default),
                ui.input_select("x_col", "Group Category", choices={"(None)": "Aggregate (All)"} | disc_choices, selected=x_default),
                class_="uf-toolbar",
            )
        return ui.p("Select variables to render this chart.", class_="text-muted small")

    @output
    @render_widget
    def plot():
        df = filtered_data()
        discrete_cols, numeric_cols = get_column_types(df)
        chart = input.chart_type()

        try:
            import statsmodels
            has_statsmodels = True
        except ImportError:
            has_statsmodels = False

        if df.empty:
            fig = px.scatter(title="No records match current filter selections. Broaden filters to display data.")
            fig.update_layout(template="plotly_white")
            return fig

        try:
            x_var = input.x_col() if hasattr(input, "x_col") else None
            y_var = input.y_col() if hasattr(input, "y_col") else None
            color_var = input.color_col() if hasattr(input, "color_col") else None
            color = color_var if (color_var and color_var != "(None)" and color_var in df.columns) else None

            # Fallback if selected input doesn't exist in current df
            if not x_var or x_var not in df.columns:
                x_var = discrete_cols[0] if discrete_cols else df.columns[0]
            if not y_var or y_var not in df.columns:
                y_var = numeric_cols[0] if numeric_cols else df.columns[-1]

            x_title = str(x_var).replace("_", " ").title()
            y_title = str(y_var).replace("_", " ").title()

            if chart == "scatter":
                fig = px.scatter(
                    df,
                    x=x_var,
                    y=y_var,
                    color=color,
                    color_discrete_sequence=UF_PALETTE,
                    trendline="ols" if (has_statsmodels and len(df) > 3 and not color) else None,
                    hover_data=df.columns.tolist()[:5],
                    labels={x_var: x_title, y_var: y_title},
                    title=f"{y_title} vs. {x_title}",
                )
            elif chart in ("bar", "line"):
                # Clean aggregation ensuring numeric y
                df_clean = df.copy()
                df_clean[y_var] = pd.to_numeric(df_clean[y_var], errors="coerce")
                df_clean = df_clean.dropna(subset=[y_var])
                
                if color:
                    grouped = df_clean.groupby([x_var, color], as_index=False)[y_var].mean()
                    if chart == "bar":
                        fig = px.bar(
                            grouped,
                            x=x_var,
                            y=y_var,
                            color=color,
                            barmode="group",
                            color_discrete_sequence=UF_PALETTE,
                            labels={x_var: x_title, y_var: y_title, color: str(color).replace("_", " ").title()},
                            title=f"Mean {y_title} by {x_title} (Stratified by {str(color).replace('_', ' ').title()})",
                        )
                    else:
                        fig = px.line(
                            grouped,
                            x=x_var,
                            y=y_var,
                            color=color,
                            markers=True,
                            color_discrete_sequence=UF_PALETTE,
                            labels={x_var: x_title, y_var: y_title, color: str(color).replace("_", " ").title()},
                            title=f"Trend of {y_title} across {x_title} (Stratified by {str(color).replace('_', ' ').title()})",
                        )
                else:
                    grouped = df_clean.groupby(x_var, as_index=False)[y_var].mean()
                    if chart == "bar":
                        fig = px.bar(
                            grouped,
                            x=x_var,
                            y=y_var,
                            color=x_var,
                            color_discrete_sequence=UF_PALETTE,
                            labels={x_var: x_title, y_var: y_title},
                            title=f"Mean {y_title} across {x_title}",
                        )
                    else:
                        fig = px.line(
                            grouped,
                            x=x_var,
                            y=y_var,
                            markers=True,
                            color_discrete_sequence=UF_PALETTE,
                            labels={x_var: x_title, y_var: y_title},
                            title=f"Mean {y_title} across {x_title}",
                        )
            elif chart == "hist":
                fig = px.histogram(
                    df,
                    x=x_var,
                    color=color,
                    marginal="box",
                    barmode="overlay" if color else "relative",
                    color_discrete_sequence=UF_PALETTE,
                    labels={x_var: x_title},
                    title=f"Distribution of {x_title}",
                )
            elif chart == "box":
                x_grp = x_var if (x_var and x_var != "(None)" and x_var in df.columns) else None
                fig = px.box(
                    df,
                    x=x_grp,
                    y=y_var,
                    points="all",
                    color=x_grp,
                    color_discrete_sequence=UF_PALETTE,
                    labels={y_var: y_title, x_grp: str(x_grp).replace("_", " ").title()} if x_grp else {y_var: y_title},
                    title=f"Distribution of {y_title}" + (f" by {str(x_grp).replace('_', ' ').title()}" if x_grp else ""),
                )
            else:
                fig = px.scatter(df, x=x_var, y=y_var, title=f"{y_title} vs. {x_title}")
        except Exception as exc:
            fig = px.scatter(title=f"Visualization notice: {exc}")

        fig.update_layout(
            template="plotly_white",
            font=dict(family="'Plus Jakarta Sans', system-ui, sans-serif", color="#0a101d", size=12),
            margin=dict(l=45, r=25, t=55, b=45),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="closest",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#edf2ff", linecolor="#dbeafe")
        fig.update_yaxes(showgrid=True, gridcolor="#edf2ff", linecolor="#dbeafe")
        return fig

        fig.update_layout(
            template="plotly_white",
            font=dict(family="'Plus Jakarta Sans', system-ui, sans-serif", color="#0a101d", size=12),
            margin=dict(l=45, r=25, t=55, b=45),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="closest",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#edf2ff", linecolor="#dbeafe")
        fig.update_yaxes(showgrid=True, gridcolor="#edf2ff", linecolor="#dbeafe")
        return fig

    @output
    @render.data_frame
    def table():
        return render.DataGrid(filtered_data().head(500), filters=True, height="380px", width="100%")

    @output
    @render.data_frame
    def manifest_table():
        return render.DataGrid(manifest(), filters=False, height="240px", width="100%")

    # Custom Upload Handlers
    @reactive.calc
    def uploaded_raw_dataset():
        files = input.upload_csv()
        if not files:
            return None
        f = files[0]
        if int(f.get("size", 0)) > 10 * 1024 * 1024:
            raise ValueError("The uploaded CSV file exceeds the 10 MB limit.")
        df = pd.read_csv(f["datapath"], encoding="utf-8-sig")
        if df.empty:
            raise ValueError("The uploaded file contains headers but no rows.")
        if len(df) > 100_000:
            raise ValueError("This tool supports up to 100,000 rows.")
        if df.columns.duplicated().any():
            raise ValueError("The file has duplicate column names. Please rename them before uploading.")
        return df

    @output
    @render.ui
    def upload_dynamic_filters():
        df = uploaded_raw_dataset()
        if df is None:
            return ui.p("Upload a CSV file to generate dynamic filters.", class_="text-muted small")
        return ui.div(*build_filter_inputs(df, prefix="upfilter_"))

    @reactive.effect
    @reactive.event(input.upload_reset)
    def _reset_upload_filters():
        df = uploaded_raw_dataset()
        if df is None:
            return
        for col in df.columns:
            if df[col].dtype == "object":
                vals = df[col].dropna().astype(str).unique()
                if 1 < len(vals) <= 50:
                    ui.update_selectize(f"upfilter_{col}", selected=[])
        for col in df.select_dtypes(include="number").columns[:3]:
            lo, hi = float(df[col].min()), float(df[col].max())
            if lo != hi:
                ui.update_slider(f"upfilter_range_{col}", value=(lo, hi))

    @reactive.calc
    def uploaded_filtered_dataset():
        df = uploaded_raw_dataset()
        if df is None:
            return None
        df = df.copy()
        for col in df.columns:
            try:
                selected = input[f"upfilter_{col}"]()
            except Exception:
                selected = []
            if selected:
                df = df[df[col].astype(str).isin(selected)]
        for col in df.select_dtypes(include="number").columns:
            try:
                value = input[f"upfilter_range_{col}"]()
            except Exception:
                value = None
            if value is not None:
                df = df[df[col].between(value[0], value[1])]
        return df

    @reactive.calc
    def uploaded_dictionary():
        files = input.upload_dictionary()
        if not files:
            return None
        return pd.read_csv(files[0]["datapath"], encoding="utf-8-sig")

    @reactive.calc
    def dictionary_check():
        df = uploaded_raw_dataset()
        dictionary = uploaded_dictionary()
        if df is None:
            return {"ok": False, "message": "Upload a CSV file to begin analysis.", "dictionary": None}
        if dictionary is None:
            return {"ok": True, "message": "File loaded successfully. (Optional dictionary omitted).", "dictionary": None}
        return validate_data_dictionary(df, dictionary)

    @output
    @render.ui
    def upload_status():
        try:
            df = uploaded_raw_dataset()
            check = dictionary_check()
        except Exception as exc:
            return ui.div(ui.strong("Notice: "), str(exc), class_="alert alert-danger", style="margin-top: 12px; font-size: 13px;")
        if df is None:
            return ui.p("Upload a file above to begin.", class_="text-muted small")
        cls = "alert alert-success" if check["ok"] else "alert alert-warning"
        return ui.div(
            ui.strong("Dataset Ready: "),
            f"{len(df):,} total rows and {len(df.columns):,} columns. ",
            check["message"],
            class_=cls,
            style="margin-top: 12px; font-size: 13px;",
        )

    @output
    @render.ui
    def upload_quality_summary():
        try:
            raw = uploaded_raw_dataset()
            filtered = uploaded_filtered_dataset()
        except Exception as exc:
            return ui.p(str(exc), class_="text-muted small")
        if raw is None:
            return ui.p("Upload a CSV file to inspect data dimensions and missingness summary.", class_="text-muted small")

        missing = int(raw.isna().sum().sum())
        duplicates = int(raw.duplicated().sum())
        cells = max(int(raw.shape[0] * raw.shape[1]), 1)
        shown_rows = len(filtered) if filtered is not None else len(raw)

        return ui.div(
            ui.div(
                ui.span("Observations (Raw / Filtered):", class_="uf-summary-lbl"),
                ui.span(f"{len(raw):,} total / {shown_rows:,} active", class_="uf-summary-val"),
                class_="uf-summary-row",
            ),
            ui.div(
                ui.span("Total Columns / Dimensions:", class_="uf-summary-lbl"),
                ui.span(f"{len(raw.columns):,}", class_="uf-summary-val"),
                class_="uf-summary-row",
            ),
            ui.div(
                ui.span("Missing Cells Rate:", class_="uf-summary-lbl"),
                ui.span(f"{missing:,} ({missing / cells:.1%})", class_="uf-summary-val"),
                class_="uf-summary-row",
            ),
            ui.div(
                ui.span("Duplicate Rows:", class_="uf-summary-lbl"),
                ui.span(f"{duplicates:,}", class_="uf-summary-val"),
                class_="uf-summary-row",
            ),
            ui.div(
                ui.span("Numeric Columns:", class_="uf-summary-lbl"),
                ui.span(f"{len(raw.select_dtypes(include='number').columns)}", class_="uf-summary-val"),
                class_="uf-summary-row",
            ),
            ui.div(
                ui.span("Categorical / Text Columns:", class_="uf-summary-lbl"),
                ui.span(f"{len(raw.select_dtypes(exclude='number').columns)}", class_="uf-summary-val"),
                class_="uf-summary-row",
            ),
            class_="uf-summary-list",
        )

    @reactive.effect
    def _update_ml_target_choices():
        try:
            df = uploaded_raw_dataset()
        except Exception:
            return
        if df is not None and len(df.columns):
            ui.update_select("ml_target", choices={str(col): str(col).replace("_", " ").title() for col in df.columns}, selected=str(df.columns[-1]))

    @reactive.calc
    @reactive.event(input.run_ml)
    def ml_result():
        if input.run_ml() == 0:
            return {"error": "Upload a dataset, choose target column, and click Train Baseline Model."}
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder

            df = uploaded_raw_dataset()
            if df is None:
                raise ValueError("Please upload a CSV dataset in the 'Upload Custom Dataset' tab first.")
            target = input.ml_target()
            if target not in df.columns:
                raise ValueError("Select a valid target column.")
            task = input.ml_task()
            data = df.dropna(subset=[target]).copy()
            if len(data) < 15:
                raise ValueError("At least 15 valid observations are required to fit a baseline model.")
            X = data.drop(columns=[target])
            if X.empty:
                raise ValueError("The dataset requires predictor columns in addition to the target.")

            if task == "classification":
                y = data[target].astype(str)
                counts = y.value_counts()
                if y.nunique() < 2:
                    raise ValueError("Classification requires at least 2 distinct outcome classes.")
                if y.nunique() > 30:
                    raise ValueError("Maximum 30 classes supported in classification baseline.")
                stratify = y if counts.min() >= 2 else None
            else:
                y = pd.to_numeric(data[target], errors="coerce")
                valid = y.notna()
                X, y = X.loc[valid], y.loc[valid]
                if len(y) < 15:
                    raise ValueError("At least 15 valid numeric target rows are required for regression.")
                stratify = None

            numeric_cols = X.select_dtypes(include="number").columns.tolist()
            categorical_cols = [col for col in X.columns if col not in numeric_cols]
            for col in categorical_cols:
                X[col] = X[col].where(X[col].notna(), "Missing").astype(str)

            transformers = []
            if numeric_cols:
                transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_cols))
            if categorical_cols:
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ])
                transformers.append(("categorical", categorical_pipeline, categorical_cols))

            if not transformers:
                raise ValueError("No usable feature columns found.")

            preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)

            if task == "classification":
                estimator = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)

            model = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            if task == "classification":
                acc = round(float(accuracy_score(y_test, predictions)), 4)
                f1 = round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4)
                metrics = [
                    {"Metric": "Prediction Accuracy", "Value": f"{acc * 100:.1f}%", "Description": "Proportion of test observations accurately predicted."},
                    {"Metric": "Balanced Accuracy", "Value": f"{round(float(balanced_accuracy_score(y_test, predictions))*100, 1)}%", "Description": "Accuracy adjusted for class imbalance."},
                    {"Metric": "Macro F1 Score", "Value": str(f1), "Description": "Harmonic mean of precision and recall across classes."},
                ]
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
                r2 = round(float(r2_score(y_test, predictions)), 4)
                mae = round(float(mean_absolute_error(y_test, predictions)), 4)
                metrics = [
                    {"Metric": "Mean Absolute Error (MAE)", "Value": str(mae), "Description": "Average absolute difference between predictions and actual values."},
                    {"Metric": "Root Mean Squared Error (RMSE)", "Value": str(round(rmse, 4)), "Description": "Square root of mean squared prediction errors."},
                    {"Metric": "R² (Variance Explained)", "Value": f"{r2 * 100:.1f}%", "Description": "Proportion of target variance explained by predictors."},
                ]

            metrics.extend([
                {"Metric": "Training Sample Size", "Value": str(len(X_train)), "Description": "Number of observations used in model fitting."},
                {"Metric": "Held-Out Test Sample", "Value": str(len(X_test)), "Description": "Number of independent observations evaluated."},
            ])
            return {"error": None, "metrics": pd.DataFrame(metrics), "task": task, "target": target}
        except Exception as exc:
            return {"error": str(exc)}

    @output
    @render.ui
    def ml_status():
        result = ml_result()
        if result.get("error"):
            return ui.div(result["error"], class_="alert alert-warning", style="font-size: 13px;")
        return ui.div(
            ui.strong("Evaluation Complete: "),
            f"Baseline {result['task'].title()} model trained on '{result['target']}'. Performance evaluated on held-out test split.",
            class_="alert alert-success",
            style="font-size: 13px;",
        )

    @output
    @render.data_frame
    def ml_metrics():
        result = ml_result()
        if result.get("error"):
            return render.DataGrid(pd.DataFrame({"Status": ["Upload a file and click Train Baseline Model to view metrics."]}), filters=False)
        return render.DataGrid(result["metrics"], filters=False, height="240px", width="100%")

    @output
    @render.download_button(filename="data_quality_summary.csv")
    def download_quality_report():
        try:
            df = uploaded_raw_dataset()
        except Exception as exc:
            yield f"error\n{str(exc)}\n"
            return
        if df is None:
            yield "message\nPlease upload a dataset first\n"
            return
        yield profile_dataframe(df).to_csv(index=False)

    @output
    @render.data_frame
    def upload_quality_table():
        try:
            df = uploaded_raw_dataset()
        except Exception as exc:
            return render.DataGrid(pd.DataFrame({"Notice": [str(exc)]}), filters=False)
        if df is None:
            return render.DataGrid(pd.DataFrame({"Notice": ["Upload a CSV file to inspect column hygiene metrics."]}), filters=False)
        return render.DataGrid(profile_dataframe(df), filters=True, height="300px", width="100%")

    @output
    @render.data_frame
    def upload_preview():
        try:
            df = uploaded_filtered_dataset()
        except Exception as exc:
            return render.DataGrid(pd.DataFrame({"Notice": [str(exc)]}), filters=False)
        if df is None:
            return render.DataGrid(pd.DataFrame({"Notice": ["Upload a CSV file to preview observation records."]}), filters=False)
        return render.DataGrid(df.head(500), filters=True, height="340px", width="100%")

    @output
    @render.ui
    def upload_chart_controls():
        try:
            df = uploaded_filtered_dataset()
        except Exception:
            return ui.p("Resolve any data issues to configure chart options.", class_="text-muted small")
        if df is None:
            return ui.p("Upload a CSV file to configure visualization options.", class_="text-muted small")

        discrete_cols, numeric_cols = get_column_types(df)
        num_choices = {c: c.replace("_", " ").title() for c in numeric_cols}
        disc_choices = {c: c.replace("_", " ").title() for c in discrete_cols}
        chart = input.upload_chart_type()
        controls = []

        if chart == "heatmap":
            if len(numeric_cols) < 2:
                return ui.p("Heatmap requires at least two numeric variables.", class_="text-muted small")
            controls.append(ui.input_selectize("upload_heatmap_cols", "Select Variables for Matrix", choices=num_choices, selected=numeric_cols[:min(8, len(numeric_cols))], multiple=True))
        elif chart == "hist":
            x_default = numeric_cols[0] if numeric_cols else df.columns[0]
            controls.append(ui.input_select("upload_x", "Distribution Variable", choices=num_choices or disc_choices, selected=x_default))
            if discrete_cols:
                controls.append(ui.input_select("upload_color", "Stratify by Group", choices={"(None)": "None"} | disc_choices, selected="(None)"))
        elif chart == "box":
            y_default = numeric_cols[0] if numeric_cols else df.columns[0]
            x_default = discrete_cols[0] if discrete_cols else "(None)"
            controls.append(ui.input_select("upload_y", "Continuous Variable", choices=num_choices or disc_choices, selected=y_default))
            if discrete_cols:
                controls.append(ui.input_select("upload_x", "Grouping Factor", choices={"(None)": "Aggregate (All)"} | disc_choices, selected=x_default))
        elif chart == "scatter":
            x_default = numeric_cols[0] if numeric_cols else df.columns[0]
            y_default = numeric_cols[1] if len(numeric_cols) > 1 else x_default
            controls.extend([
                ui.input_select("upload_x", "X Variable", choices=num_choices or disc_choices, selected=x_default),
                ui.input_select("upload_y", "Y Variable", choices=num_choices or disc_choices, selected=y_default),
            ])
            if discrete_cols:
                controls.append(ui.input_select("upload_color", "Color Grouping", choices={"(None)": "None"} | disc_choices, selected="(None)"))
        elif chart in ("bar", "line"):
            x_default = discrete_cols[0] if discrete_cols else df.columns[0]
            y_default = numeric_cols[0] if numeric_cols else df.columns[-1]
            controls.extend([
                ui.input_select("upload_x", "Category Dimension", choices=disc_choices or num_choices, selected=x_default),
                ui.input_select("upload_y", "Aggregated Measure (Mean)", choices=num_choices or disc_choices, selected=y_default),
            ])
            if len(discrete_cols) > 1:
                sub_cats = {"(None)": "None"} | {c: disc_choices[c] for c in discrete_cols if c != x_default}
                controls.append(ui.input_select("upload_color", "Sub-Group (Color)", choices=sub_cats, selected="(None)"))
        return ui.div(*controls, class_="uf-toolbar")

    @output
    @render_widget
    def upload_plot():
        try:
            df = uploaded_filtered_dataset()
        except Exception as exc:
            return px.scatter(title=f"Notice: {exc}")
        if df is None:
            return px.scatter(title="Upload a CSV file to render visualization.")
        if df.empty:
            return px.scatter(title="No records match current filter criteria.")

        chart = input.upload_chart_type()
        discrete_cols, numeric_cols = get_column_types(df)

        try:
            if chart == "heatmap" and len(numeric_cols) >= 2:
                selected = input.upload_heatmap_cols()
                selected = [c for c in selected if c in numeric_cols] if selected else numeric_cols[:8]
                if len(selected) < 2:
                    selected = numeric_cols[:min(8, len(numeric_cols))]
                corr = df[selected].corr(numeric_only=True)
                labels = [c.replace("_", " ").title() for c in selected]
                fig = px.imshow(
                    corr,
                    x=labels,
                    y=labels,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Matrix",
                )
            elif chart == "hist":
                x = input.upload_x() if hasattr(input, "upload_x") and input.upload_x() in df.columns else (numeric_cols[0] if numeric_cols else df.columns[0])
                x_title = str(x).replace("_", " ").title()
                try:
                    color = input.upload_color() if input.upload_color() != "(None)" else None
                except Exception:
                    color = None
                fig = px.histogram(
                    df,
                    x=x,
                    color=color,
                    marginal="box",
                    barmode="overlay" if color else "relative",
                    color_discrete_sequence=UF_PALETTE,
                    labels={x: x_title},
                    title=f"Distribution of {x_title}",
                )
            elif chart == "box":
                y = input.upload_y() if hasattr(input, "upload_y") and input.upload_y() in df.columns else (numeric_cols[0] if numeric_cols else df.columns[0])
                y_title = str(y).replace("_", " ").title()
                try:
                    x = input.upload_x() if input.upload_x() != "(None)" and input.upload_x() in df.columns else None
                except Exception:
                    x = None
                fig = px.box(
                    df,
                    x=x,
                    y=y,
                    points="outliers",
                    color=x,
                    color_discrete_sequence=UF_PALETTE,
                    labels={y: y_title, x: str(x).replace("_", " ").title()} if x else {y: y_title},
                    title=f"Distribution of {y_title}" + (f" across {str(x).replace('_', ' ').title()}" if x else ""),
                )
            elif chart == "scatter":
                x = input.upload_x() if hasattr(input, "upload_x") and input.upload_x() in df.columns else (numeric_cols[0] if numeric_cols else df.columns[0])
                y = input.upload_y() if hasattr(input, "upload_y") and input.upload_y() in df.columns else (numeric_cols[1] if len(numeric_cols) > 1 else x)
                x_title, y_title = str(x).replace("_", " ").title(), str(y).replace("_", " ").title()
                try:
                    color = input.upload_color() if input.upload_color() != "(None)" else None
                except Exception:
                    color = None
                try:
                    import statsmodels
                    has_sm = True
                except ImportError:
                    has_sm = False
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    color=color,
                    trendline="ols" if (has_sm and len(df) > 3 and not color) else None,
                    color_discrete_sequence=UF_PALETTE,
                    labels={x: x_title, y: y_title},
                    title=f"{y_title} vs. {x_title}",
                )
            elif chart in ("bar", "line"):
                x = input.upload_x() if hasattr(input, "upload_x") and input.upload_x() in df.columns else (discrete_cols[0] if discrete_cols else df.columns[0])
                y = input.upload_y() if hasattr(input, "upload_y") and input.upload_y() in df.columns else (numeric_cols[0] if numeric_cols else df.columns[-1])
                x_title, y_title = str(x).replace("_", " ").title(), str(y).replace("_", " ").title()
                try:
                    color = input.upload_color() if input.upload_color() != "(None)" else None
                except Exception:
                    color = None

                df_clean = df.copy()
                df_clean[y] = pd.to_numeric(df_clean[y], errors="coerce")
                df_clean = df_clean.dropna(subset=[y])

                if chart == "bar":
                    if color:
                        grouped = df_clean.groupby([x, color], dropna=False, as_index=False)[y].mean().sort_values(y, ascending=False).head(40)
                        fig = px.bar(grouped, x=x, y=y, color=color, barmode="group", color_discrete_sequence=UF_PALETTE, labels={x: x_title, y: y_title}, title=f"Mean {y_title} by {x_title} (Stratified by {str(color).replace('_', ' ').title()})")
                    else:
                        grouped = df_clean.groupby(x, dropna=False, as_index=False)[y].mean().sort_values(y, ascending=False).head(30)
                        fig = px.bar(grouped, x=x, y=y, color=x, color_discrete_sequence=UF_PALETTE, labels={x: x_title, y: y_title}, title=f"Mean {y_title} by {x_title} (Top Groups)")
                else:
                    if color:
                        grouped = df_clean.groupby([x, color], dropna=False, as_index=False)[y].mean()
                        fig = px.line(grouped, x=x, y=y, color=color, markers=True, color_discrete_sequence=UF_PALETTE, labels={x: x_title, y: y_title}, title=f"Trend of {y_title} by {x_title}")
                    else:
                        grouped = df_clean.groupby(x, dropna=False, as_index=False)[y].mean()
                        fig = px.line(grouped, x=x, y=y, markers=True, color_discrete_sequence=UF_PALETTE, labels={x: x_title, y: y_title}, title=f"Trend of Mean {y_title} across {x_title}")
            else:
                fig = px.scatter(title="Select compatible variables for this visualization.")
        except Exception as exc:
            fig = px.scatter(title=f"Could not render chart: {exc}")

        fig.update_layout(
            template="plotly_white",
            font=dict(family="'Plus Jakarta Sans', system-ui, sans-serif", color="#0a101d", size=12),
            margin=dict(l=45, r=25, t=55, b=45),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig


app = App(app_ui, server, static_assets=BASE_DIR / "www")
