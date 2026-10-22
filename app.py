from pathlib import Path
import pandas as pd
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
    ui.div(
        ui.div(ui.div("CUSTOM DATA SCIENCE WORKSPACE", class_="sl"),
               ui.h2("Analyze your own dataset", class_="pt"),
               ui.p("Upload a CSV and its data dictionary to run a quality audit and explore additional chart types.", class_="ps"), class_="panel"),
        ui.layout_columns(
            ui.card(
                ui.card_header("1. Upload files"),
                ui.input_file("upload_csv", "Dataset (CSV, max 10 MB)", accept=[".csv"], multiple=False),
                ui.input_file("upload_dictionary", "Data dictionary (CSV)", accept=[".csv"], multiple=False),
                ui.tags.p("Required dictionary columns: column_name, description, data_type. Optional: required, allowed_values.", class_="muted"),
                ui.tags.a("Download data dictionary template", href="/data_dictionary_template.csv", download="data_dictionary_template.csv"),
                ui.output_ui("upload_status"),
                ui.download_button("download_quality_report", "Download quality profile CSV"),
            ),
            ui.card(
                ui.card_header("2. Explore uploaded data"),
                ui.input_select("upload_chart_type", "Visualization", choices={"hist":"Histogram", "box":"Box plot", "scatter":"Scatter plot", "bar":"Bar chart", "heatmap":"Correlation heatmap"}, selected="hist"),
                ui.output_ui("upload_chart_controls"),
                ui.output_ui("upload_quality_summary"),
            ),
            col_widths=(5, 7),
        ),
        ui.div(
            ui.h3("Baseline machine learning lab", class_="pt"),
            ui.p("Train a leakage-aware baseline model on the uploaded dataset. Choose the target column and task; the app uses a held-out test split and preprocessing pipeline.", class_="ps"),
            ui.layout_columns(
                ui.input_select("ml_target", "Target column", choices={}),
                ui.input_select("ml_task", "Prediction task", choices={"classification": "Classification", "regression": "Regression"}, selected="classification"),
                ui.input_action_button("run_ml", "Train baseline model"),
                col_widths=(5, 4, 3),
            ),
            ui.output_ui("ml_status"),
            ui.output_data_frame("ml_metrics"),
            class_="panel",
        ),
        ui.div(ui.h3("Data quality profile", class_="pt"), ui.output_data_frame("upload_quality_table"), class_="panel"),
        ui.div(ui.h3("Preview uploaded data", class_="pt"), ui.output_data_frame("upload_preview"), class_="panel"),
        ui.div(output_widget("upload_plot"), class_="panel"),
        class_="content",
    ),
    ui.div("University of Florida • Interactive data science prototype",class_="foot")
)


def server(input, output, session):
    @reactive.calc
    def manifest(): return load_manifest()

    @reactive.effect
    def _update_choices():
        m=manifest(); choices=dict(zip(m.dataset_id,m.dataset_name)); ui.update_select("dataset",choices=choices,selected=m.iloc[0].dataset_id)

    @reactive.calc
    def raw_data(): return load_dataset(input.dataset())

    @reactive.effect
    @reactive.event(input.reset)
    def _reset_filters():
        df = raw_data()
        for col in df.columns:
            if df[col].dtype == "object":
                vals = df[col].dropna().astype(str).unique()
                if 1 < len(vals) <= 30:
                    ui.update_selectize(f"filter_{col}", selected=[])
        for col in df.select_dtypes(include="number").columns[:2]:
            lo, hi = float(df[col].min()), float(df[col].max())
            if lo != hi:
                ui.update_slider(f"range_{col}", value=(lo, hi))

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

    @reactive.calc
    def uploaded_dataset():
        files = input.upload_csv()
        if not files:
            return None
        f = files[0]
        if int(f.get("size", 0)) > 10 * 1024 * 1024:
            raise ValueError("The uploaded CSV exceeds the 10 MB limit. Please use a smaller file.")
        df = pd.read_csv(f["datapath"], encoding="utf-8-sig")
        if df.empty:
            raise ValueError("The uploaded CSV has headers but no data rows.")
        if len(df) > 100_000:
            raise ValueError("This demo supports up to 100,000 rows per uploaded dataset.")
        if len(df.columns) > 150:
            raise ValueError("This demo supports up to 150 columns per uploaded dataset.")
        if df.columns.duplicated().any():
            raise ValueError("The CSV contains duplicate column names. Rename them before uploading.")
        return df

    @reactive.calc
    def uploaded_dictionary():
        files = input.upload_dictionary()
        if not files:
            return None
        return pd.read_csv(files[0]["datapath"], encoding="utf-8-sig")

    @reactive.calc
    def dictionary_check():
        df = uploaded_dataset()
        dictionary = uploaded_dictionary()
        if df is None:
            return {"ok": False, "message": "Upload a CSV dataset to begin.", "dictionary": None}
        if dictionary is None:
            return {"ok": False, "message": "A data dictionary is required. Download the template, fill it in, and upload it.", "dictionary": None}
        return validate_data_dictionary(df, dictionary)

    @output
    @render.ui
    def upload_status():
        try:
            df = uploaded_dataset()
            check = dictionary_check()
        except Exception as exc:
            return ui.div(ui.strong("Upload issue: "), str(exc), class_="alert alert-danger")
        if df is None:
            return ui.p("Your uploaded data stays in this app session; it is not added to the built-in datasets.", class_="muted")
        cls = "alert alert-success" if check["ok"] else "alert alert-warning"
        return ui.div(ui.strong("Dataset loaded: "), f"{len(df):,} rows × {len(df.columns):,} columns. ", check["message"], class_=cls)

    @output
    @render.ui
    def upload_quality_summary():
        try:
            df = uploaded_dataset()
        except Exception as exc:
            return ui.p(str(exc), class_="muted")
        if df is None:
            return ui.p("Upload a dataset to see quality metrics.", class_="muted")
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        cells = max(int(df.shape[0] * df.shape[1]), 1)
        return ui.div(
            ui.p(f"Missing cells: {missing:,} ({missing / cells:.1%})"),
            ui.p(f"Duplicate rows: {duplicates:,}"),
            ui.p(f"Numeric columns: {len(df.select_dtypes(include='number').columns)}"),
            ui.p(f"Categorical/text columns: {len(df.select_dtypes(exclude='number').columns)}"),
        )

    @reactive.effect
    def _update_ml_target_choices():
        try:
            df = uploaded_dataset()
        except Exception:
            return
        if df is not None and len(df.columns):
            ui.update_select("ml_target", choices={str(col): str(col) for col in df.columns}, selected=str(df.columns[-1]))

    @reactive.calc
    @reactive.event(input.run_ml)
    def ml_result():
        if input.run_ml() == 0:
            return {"error": "Upload a dataset, choose a target, and click Train baseline model."}
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder
            import numpy as np

            df = uploaded_dataset()
            if df is None:
                raise ValueError("Upload a CSV dataset first.")
            target = input.ml_target()
            if target not in df.columns:
                raise ValueError("Choose a target column from the uploaded dataset.")
            task = input.ml_task()
            data = df.dropna(subset=[target]).copy()
            if len(data) < 20:
                raise ValueError("At least 20 usable rows are required for a baseline experiment.")
            X = data.drop(columns=[target])
            if X.empty:
                raise ValueError("The dataset needs at least one feature column besides the target.")

            if task == "classification":
                y = data[target].astype(str)
                counts = y.value_counts()
                if y.nunique() < 2:
                    raise ValueError("Classification requires at least two target classes.")
                if y.nunique() > 30:
                    raise ValueError("This classification demo supports up to 30 classes. Choose another target or use regression.")
                if counts.min() < 2:
                    raise ValueError("Each target class needs at least two observations for a train/test split.")
                stratify = y if counts.min() >= 2 else None
            else:
                y = pd.to_numeric(data[target], errors="coerce")
                valid = y.notna()
                X, y = X.loc[valid], y.loc[valid]
                if len(y) < 20:
                    raise ValueError("At least 20 numeric target observations are required for regression.")
                if y.nunique() < 2:
                    raise ValueError("Regression requires a target with more than one distinct value.")
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
                raise ValueError("No usable feature columns were found.")
            preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
            if task == "classification":
                estimator = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
            else:
                estimator = RandomForestRegressor(n_estimators=150, random_state=42)
            model = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            if task == "classification":
                metrics = [
                    {"metric": "Accuracy", "value": round(float(accuracy_score(y_test, predictions)), 4)},
                    {"metric": "Balanced accuracy", "value": round(float(balanced_accuracy_score(y_test, predictions)), 4)},
                    {"metric": "Macro F1", "value": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4)},
                ]
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
                metrics = [
                    {"metric": "MAE", "value": round(float(mean_absolute_error(y_test, predictions)), 4)},
                    {"metric": "RMSE", "value": round(rmse, 4)},
                    {"metric": "R²", "value": round(float(r2_score(y_test, predictions)), 4)},
                ]
            metrics.extend([
                {"metric": "Training rows", "value": int(len(X_train))},
                {"metric": "Test rows", "value": int(len(X_test))},
                {"metric": "Feature columns", "value": int(X.shape[1])},
            ])
            return {"error": None, "metrics": pd.DataFrame(metrics), "task": task, "target": target}
        except Exception as exc:
            return {"error": str(exc)}

    @output
    @render.ui
    def ml_status():
        result = ml_result()
        if result.get("error"):
            return ui.div(result["error"], class_="alert alert-warning")
        return ui.div(
            ui.strong("Baseline complete. "),
            f"{result['task'].title()} model for target '{result['target']}'. These are held-out test metrics, not a guarantee of real-world performance.",
            class_="alert alert-success",
        )

    @output
    @render.data_frame
    def ml_metrics():
        result = ml_result()
        if result.get("error"):
            return render.DataGrid(pd.DataFrame({"Status": ["Train a model to view metrics."]}), filters=False)
        return render.DataGrid(result["metrics"], filters=False, height="280px", width="100%")

    @output
    @render.download_button(filename="data_quality_profile.csv")
    def download_quality_report():
        try:
            df = uploaded_dataset()
        except Exception as exc:
            yield f"error\n{str(exc)}\n"
            return
        if df is None:
            yield "message\nUpload a dataset first\n"
            return
        yield profile_dataframe(df).to_csv(index=False)

    @output
    @render.data_frame
    def upload_quality_table():
        try:
            df = uploaded_dataset()
        except Exception as exc:
            return render.DataGrid(pd.DataFrame({"Upload issue": [str(exc)]}), filters=False)
        if df is None:
            return render.DataGrid(pd.DataFrame({"Message": ["Upload a CSV and data dictionary to see the quality profile."]}), filters=False)
        return render.DataGrid(profile_dataframe(df), filters=True, height="360px", width="100%")

    @output
    @render.data_frame
    def upload_preview():
        try:
            df = uploaded_dataset()
        except Exception as exc:
            return render.DataGrid(pd.DataFrame({"Upload issue": [str(exc)]}), filters=False)
        if df is None:
            return render.DataGrid(pd.DataFrame({"Message": ["Upload a CSV to preview it here."]}), filters=False)
        return render.DataGrid(df.head(500), filters=True, height="360px", width="100%")

    @output
    @render.ui
    def upload_chart_controls():
        try:
            df = uploaded_dataset()
        except Exception:
            return ui.p("Fix the upload issue to enable chart controls.", class_="muted")
        if df is None:
            return ui.p("Upload a CSV to enable chart controls.", class_="muted")
        numeric = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()
        categorical = [c for c in all_cols if c not in numeric]
        chart = input.upload_chart_type()
        controls = []
        if chart == "heatmap":
            if len(numeric) < 2:
                return ui.p("A correlation heatmap requires at least two numeric columns.", class_="muted")
            controls.append(ui.input_selectize("upload_heatmap_cols", "Numeric variables", choices=numeric, selected=numeric[:min(8, len(numeric))], multiple=True))
        elif chart == "hist":
            if not numeric: return ui.p("A histogram requires a numeric column.", class_="muted")
            controls.append(ui.input_select("upload_x", "Numeric variable", choices=numeric, selected=numeric[0]))
        elif chart == "box":
            if not numeric: return ui.p("A box plot requires a numeric column.", class_="muted")
            controls.append(ui.input_select("upload_y", "Numeric variable", choices=numeric, selected=numeric[0]))
            if categorical: controls.append(ui.input_select("upload_x", "Optional grouping variable", choices=["(none)"] + categorical, selected="(none)"))
        elif chart == "scatter":
            if len(numeric) < 2: return ui.p("A scatter plot requires at least two numeric columns.", class_="muted")
            controls.extend([ui.input_select("upload_x", "X variable", choices=numeric, selected=numeric[0]), ui.input_select("upload_y", "Y variable", choices=numeric, selected=numeric[1])])
        elif chart == "bar":
            if not categorical or not numeric: return ui.p("A bar chart requires a categorical column and a numeric column.", class_="muted")
            controls.extend([ui.input_select("upload_x", "Category", choices=categorical, selected=categorical[0]), ui.input_select("upload_y", "Numeric measure (mean by category)", choices=numeric, selected=numeric[0])])
        return ui.div(*controls, class_="cc")

    @output
    @render_widget
    def upload_plot():
        try:
            df = uploaded_dataset()
        except Exception as exc:
            return px.scatter(title=f"Upload issue: {exc}")
        if df is None:
            return px.scatter(title="Upload a CSV to start exploring.")
        if df.empty:
            return px.scatter(title="No rows available to visualize.")
        chart = input.upload_chart_type()
        numeric = df.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in df.columns if c not in numeric]
        try:
            if chart == "heatmap" and len(numeric) >= 2:
                selected = input.upload_heatmap_cols()
                selected = [c for c in selected if c in numeric] if selected else numeric[:8]
                if len(selected) < 2: selected = numeric[:min(8, len(numeric))]
                fig = px.imshow(df[selected].corr(numeric_only=True), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric correlation matrix")
            elif chart == "hist" and numeric:
                x = input.upload_x()
                fig = px.histogram(df, x=x, marginal="box", title=f"Distribution of {x}")
            elif chart == "box" and numeric:
                y = input.upload_y()
                try:
                    x = input.upload_x()
                except Exception:
                    x = "(none)"
                if x and x != "(none)" and x in categorical:
                    fig = px.box(df, x=x, y=y, points="outliers", title=f"{y} by {x}")
                else:
                    fig = px.box(df, y=y, points="outliers", title=f"Distribution of {y}")
            elif chart == "scatter" and len(numeric) >= 2:
                x, y = input.upload_x(), input.upload_y()
                fig = px.scatter(df, x=x, y=y, title=f"{y} versus {x}")
            elif chart == "bar" and categorical and numeric:
                x, y = input.upload_x(), input.upload_y()
                grouped = df.groupby(x, dropna=False, as_index=False)[y].mean().sort_values(y, ascending=False).head(50)
                fig = px.bar(grouped, x=x, y=y, title=f"Mean {y} by {x} (top 50 groups)")
            else:
                fig = px.scatter(title="Choose compatible variables for this chart.")
        except Exception as exc:
            fig = px.scatter(title=f"Could not render chart: {exc}")
        fig.update_layout(template="plotly_white", margin=dict(l=35, r=20, t=65, b=45), paper_bgcolor="white", plot_bgcolor="white")
        return fig

app=App(app_ui,server,static_assets=BASE_DIR/"www")
