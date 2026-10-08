# UF Political Science Data Lab — Shiny for Python

Standalone prototype for an interactive political-science visualization lab.

## Run in VS Code PowerShell

```powershell
py -m pip install -r requirements.txt
py -m shiny run --launch-browser app.py
```

If `py` is unavailable, use:

```powershell
python -m pip install -r requirements.txt
python -m shiny run --launch-browser app.py
```

## Dynamic datasets

Datasets live in `data/`. `manifest.csv` is the registry. To add a new CSV, put it in `data/` and add one row to `manifest.csv`. The app discovers the file at runtime. Paths are resolved relative to `app.py`, so launching from VS Code does not break dataset loading.

The included datasets are synthetic/prototype classroom data. Replace them with documented real sources for production use.
