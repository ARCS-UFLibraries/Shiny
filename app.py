from pathlib import Path
import pandas as pd
import plotly.express as px
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"


def load_manifest():
    return pd.read_csv(MANIFEST_PATH)


def load_dataset(dataset_id):
    m = load_manifest()
    row = m.loc[m.dataset_id == dataset_id]
    if row.empty:
        raise ValueError(f"Dataset '{dataset_id}' is not registered in manifest.csv")
    path = DATA_DIR / str(row.iloc[0]["file"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    return pd.read_csv(path)


def dataset_info(dataset_id):
    return load_manifest().loc[lambda x: x.dataset_id == dataset_id].iloc[0]


def make_filters(df):
    controls = []
    cats = [c for c in df.columns if df[c].dtype == "object"]
    for col in cats[:4]:
        vals = sorted(df[col].dropna().astype(str).unique())
        if 1 < len(vals) <= 30:
            controls.append(ui.input_selectize(f"filter_{col}", col.replace("_", " ").title(), choices=vals, multiple=True, options={"placeholder":"All values"}))
    nums = df.select_dtypes(include="number").columns.tolist()
    for col in nums[:2]:
        lo, hi = float(df[col].min()), float(df[col].max())
        if lo != hi:
            controls.append(ui.input_slider(f"range_{col}", col.replace("_", " ").title(), min=lo, max=hi, value=(lo, hi)))
    return controls or [ui.p("No filters are needed for this dataset.", class_="muted")]


CSS = """
:root{--blue:#0021A5;--orange:#FA4616;--ink:#172033;--muted:#667085;--line:#e6eaf0;--bg:#f5f7fb}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Arial,sans-serif}
.topbar{height:72px;background:var(--blue);color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 42px;box-shadow:0 2px 12px #0021a52e}.brand{display:flex;align-items:center;gap:14px}.uf{width:42px;height:42px;border-radius:10px;background:#fff;color:var(--blue);display:flex;align-items:center;justify-content:center;font-weight:800}.bt{font-size:18px;font-weight:700}.bs{font-size:11px;opacity:.72}.pill{border:1px solid #ffffff59;border-radius:999px;padding:7px 11px;font-size:10px;font-weight:700}
.hero{max-width:1440px;margin:28px auto 24px;background:#fff;border-radius:18px;border:1px solid var(--line);display:grid;grid-template-columns:1.7fr 1fr;overflow:hidden;box-shadow:0 8px 30px #1720330d}.hc{padding:42px 46px}.ht{font-size:38px;line-height:1.05;letter-spacing:-1.5px;margin:0 0 12px;color:var(--blue)}.hx{font-size:16px;color:var(--muted);max-width:680px;line-height:1.65}.hs{background:linear-gradient(135deg,var(--blue),#00166f);color:#fff;padding:38px 40px}.hk,.sl,.rl{font-size:10px;font-weight:800;letter-spacing:1.1px;color:var(--orange)}.hk{color:#ffb39d}.hst{font-size:24px;font-weight:700;margin:12px 0 8px}.hsx{opacity:.82;line-height:1.6;font-size:13px}.steps{display:flex;align-items:center;gap:8px;margin-top:28px;font-size:12px;font-weight:600}.sn{width:25px;height:25px;border-radius:50%;background:#eef2ff;color:var(--blue);display:inline-flex;align-items:center;justify-content:center;font-size:11px}.sa{color:#a0a8b8}
.content{max-width:1440px;margin:auto}.bslib-sidebar-layout{background:transparent!important}.bslib-sidebar-panel{background:#fff!important;border:1px solid var(--line)!important;border-radius:16px!important;padding:22px!important;box-shadow:0 6px 24px #1720330a}.sl{margin-bottom:9px}.muted{color:var(--muted);font-size:12px;line-height:1.55}.form-label{font-size:11px;font-weight:700;color:#4d5769}.form-select,.selectize-input{border-color:#dfe4ec!important;border-radius:9px!important;min-height:40px!important}.reset-btn{background:#fff!important;color:var(--blue)!important;border:1px solid #ccd4e3!important;border-radius:9px!important;font-weight:600!important}
.krow{display:flex;gap:14px;margin:20px 0}.k{flex:1;background:#fff;border:1px solid var(--line);border-radius:14px;padding:17px 20px}.kw{flex:2}.kv{font-size:25px;font-weight:800;color:var(--blue)}.ks{font-size:18px}.kl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-top:4px}.panel{background:#fff;border:1px solid var(--line);border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 6px 24px #1720330a}.ph{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.pt{margin:4px 0;font-size:21px}.ps{margin:4px 0;color:var(--muted);font-size:12px}.cs{min-width:160px}.cc{display:flex;gap:12px;margin:18px 0 8px;padding:12px;background:#f8f9fc;border-radius:10px}.cc .form-group{flex:1}.plot{min-height:440px}.reflection{border-left:4px solid var(--orange);background:#fff8f5;padding:17px 20px;border-radius:0 10px 10px 0;margin-top:10px}.rt{margin:4px 0;font-size:16px}.rx{margin:0;color:var(--muted);font-size:12px;line-height:1.6}.arow{display:grid;grid-template-columns:auto 1fr auto 1fr auto 1fr;gap:12px;align-items:center;margin-top:20px}.an{width:30px;height:30px;border-radius:50%;background:var(--blue);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px}.arow strong{font-size:12px}.arow p{color:var(--muted);font-size:11px;margin:3px 0}.foot{text-align:center;padding:28px;color:#8a93a5;font-size:10px}@media(max-width:900px){.hero{grid-template-columns:1fr}.krow{flex-wrap:wrap}.k,.kw{flex-basis:30%}.arow{grid-template-columns:auto 1fr}}
"""

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.title("UF Political Science Data Lab"), ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"), ui.tags.style(CSS)),
    ui.div(ui.div(ui.div("UF", class_="uf"), ui.div(ui.div("Political Science Data Lab", class_="bt"), ui.div("Interactive Data Visualization Learning Environment", class_="bs")), class_="brand"), ui.span("PYTHON • SHINY", class_="pill"), class_="topbar"),
    ui.div(ui.div(ui.div("EXPLORE • FILTER • INTERPRET", class_="hk"), ui.h1("Explore political data.", class_="ht"), ui.p("Filter a dataset, change the visualization, and investigate what the data is telling you.", class_="hx"), ui.div(ui.span("1",class_="sn"),ui.span("Choose data"),ui.span("→",class_="sa"),ui.span("2",class_="sn"),ui.span("Explore"),ui.span("→",class_="sa"),ui.span("3",class_="sn"),ui.span("Interpret"),class_="steps"),class_="hc"), ui.div(ui.div("DATA-DRIVEN LEARNING",class_="hk"),ui.div("Built to grow as new datasets arrive.",class_="hst"),ui.p("Datasets are registered separately from the application, so instructors can add or update data without rebuilding the interface.",class_="hsx"),class_="hs"),class_="hero"),
    ui.layout_sidebar(ui.sidebar(ui.div("DATASET",class_="sl"),ui.input_select("dataset",None,choices={}),ui.output_ui("dataset_description"),ui.hr(),ui.div("FILTERS",class_="sl"),ui.output_ui("dynamic_filters"),ui.hr(),ui.input_action_button("reset","Reset filters",class_="reset-btn",width="100%"),open="open",width="310px"),
        ui.div(ui.div(ui.div(ui.output_text("row_kpi"),class_="kv"),ui.div("Rows shown",class_="kl"),class_="k"),ui.div(ui.div(ui.output_text("column_kpi"),class_="kv"),ui.div("Variables",class_="kl"),class_="k"),ui.div(ui.div(ui.output_text("dataset_kpi"),class_="kv ks"),ui.div("Current dataset",class_="kl"),class_="k kw"),class_="krow"),
        ui.div(ui.div(ui.div(ui.div("VISUAL EXPLORER",class_="sl"),ui.h2("Build a view of the data",class_="pt")),ui.input_select("chart_type",None,choices={"bar":"Bar chart","line":"Line chart","scatter":"Scatter plot","hist":"Distribution"},selected="bar"),class_="ph"),ui.output_ui("chart_controls"),ui.div(output_widget("plot"),class_="plot"),ui.div(ui.div("THINK LIKE A DATA ANALYST",class_="rl"),ui.h3("What do you notice?",class_="rt"),ui.p("Describe one pattern, difference, or relationship you can see. Then ask what additional information you would need before making a political claim.",class_="rx"),class_="reflection"),class_="panel"),
        ui.div(ui.div(ui.div("DATA PREVIEW",class_="sl"),ui.h2("See the rows behind the chart",class_="pt"),ui.p("The table updates with the same filters used by the visualization.",class_="ps"),class_="ph"),ui.output_data_frame("table"),class_="panel"),
        ui.div(ui.div(ui.div("DESIGN NOTE",class_="sl"),ui.h2("Why the data layer is separate",class_="pt"),ui.p("A small registry tells the app which CSV files exist. Adding a dataset means adding a file and one registry row—not rewriting the visualization interface.",class_="ps"),ui.div(ui.div("1",class_="an"),ui.div(ui.strong("Add the dataset"),ui.p("Place a new CSV in data/.")),ui.div("2",class_="an"),ui.div(ui.strong("Register it"),ui.p("Add one row to manifest.csv.")),ui.div("3",class_="an"),ui.div(ui.strong("Explore it"),ui.p("The same interface loads it.")),class_="arow"),class_="panel"),class_="content"),
    ),
    ui.div("University of Florida • Standalone instructional prototype",class_="foot")
)


def server(input, output, session):
    @reactive.calc
    def manifest(): return load_manifest()

    @reactive.effect
    def _update_choices():
        m=manifest(); choices=dict(zip(m.dataset_id,m.dataset_name)); ui.update_select("dataset",choices=choices,selected=m.iloc[0].dataset_id)

    @reactive.calc
    def raw_data(): return load_dataset(input.dataset())

    @reactive.calc
    def filtered_data():
        df=raw_data().copy()
        for col in df.columns:
            try: selected=input[f"filter_{col}"]()
            except Exception: selected=[]
            if selected: df=df[df[col].astype(str).isin(selected)]
        for col in df.select_dtypes(include="number").columns:
            try: value=input[f"range_{col}"]()
            except Exception: value=None
            if value is not None: df=df[df[col].between(value[0],value[1])]
        return df

    @output
    @render.ui
    def dataset_description():
        r=dataset_info(input.dataset()); return ui.div(ui.p(r.description,class_="muted"),ui.p(ui.strong("Source: "),r.source,class_="muted"))

    @output
    @render.ui
    def dynamic_filters(): return ui.div(*make_filters(raw_data()))

    @output
    @render.text
    def row_kpi(): return f"{len(filtered_data()):,}"
    @output
    @render.text
    def column_kpi(): return str(len(filtered_data().columns))
    @output
    @render.text
    def dataset_kpi(): return str(dataset_info(input.dataset()).dataset_name)

    @output
    @render.ui
    def chart_controls():
        df=filtered_data(); nums=df.select_dtypes(include="number").columns.tolist(); cats=[c for c in df.columns if c not in nums]; chart=input.chart_type()
        if chart=="scatter" and len(nums)>=2: return ui.div(ui.input_select("x_col","X variable",choices=nums,selected=nums[0]),ui.input_select("y_col","Y variable",choices=nums,selected=nums[1]),class_="cc")
        if chart in ("bar","line") and cats and nums: return ui.div(ui.input_select("x_col","Category",choices=cats,selected=cats[0]),ui.input_select("y_col","Measure",choices=nums,selected=nums[0]),class_="cc")
        if chart=="hist" and nums: return ui.div(ui.input_select("x_col","Variable",choices=nums,selected=nums[0]),class_="cc")
        return ui.p("Choose compatible variables for this visualization.",class_="muted")

    @output
    @render_widget
    def plot():
        df=filtered_data(); nums=df.select_dtypes(include="number").columns.tolist(); cats=[c for c in df.columns if c not in nums]; chart=input.chart_type()
        if df.empty: fig=px.scatter(title="No rows match the current filters.")
        elif chart=="scatter" and len(nums)>=2: fig=px.scatter(df,x=input.x_col(),y=input.y_col(),color_discrete_sequence=["#0021A5"],title="Relationship between selected variables")
        elif chart in ("bar","line") and cats and nums:
            x,y=input.x_col(),input.y_col()
            if chart=="bar": fig=px.bar(df,x=x,y=y,color_discrete_sequence=["#0021A5"],title=f"{y.replace('_',' ').title()} by {x.replace('_',' ').title()}")
            else: fig=px.line(df.groupby(x,as_index=False)[y].mean(),x=x,y=y,markers=True,color_discrete_sequence=["#FA4616"],title=f"Average {y.replace('_',' ').title()} by {x.replace('_',' ').title()}")
        elif chart=="hist" and nums: fig=px.histogram(df,x=input.x_col(),color_discrete_sequence=["#0021A5"],title=f"Distribution of {input.x_col().replace('_',' ').title()}")
        else: fig=px.scatter(title="Choose compatible variables.")
        fig.update_layout(template="plotly_white",font=dict(family="Inter, Arial",color="#172033"),margin=dict(l=45,r=20,t=60,b=45),paper_bgcolor="white",plot_bgcolor="white")
        fig.update_xaxes(showgrid=True,gridcolor="#edf0f5"); fig.update_yaxes(showgrid=True,gridcolor="#edf0f5")
        return fig

    @output
    @render.data_frame
    def table(): return render.DataGrid(filtered_data().head(500),filters=True,height="430px",width="100%")

app=App(app_ui,server,static_assets=BASE_DIR/"www")
