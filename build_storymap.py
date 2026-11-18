import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WWW_DIR = BASE_DIR / "www"
WWW_DIR.mkdir(exist_ok=True)

# 1. Load US Elections Data
df_us = pd.read_csv(DATA_DIR / "us_elections.csv")
us_data = {}
for _, row in df_us.iterrows():
    st = row["state"]
    yr = int(row["year"])
    if st not in us_data:
        us_data[st] = {}
    us_data[st][yr] = {
        "turnout": round(float(row["turnout_percent"]), 1),
        "young_turnout": round(float(row["young_voter_turnout"]), 1),
        "mail_ballot": round(float(row["mail_ballot_percent"]), 1),
        "region": str(row["region"]),
    }

# Centroids for US States in dataset
us_coords = {
    "Florida": [27.9944, -81.7603],
    "Georgia": [32.1656, -82.9001],
    "Ohio": [40.4173, -82.9071],
    "Michigan": [44.3148, -85.6024],
    "California": [36.7783, -119.4179],
    "Texas": [31.9686, -99.9018],
}

# 2. Load ESS Trends Data
df_ess = pd.read_csv(DATA_DIR / "european_social_survey_trends.csv")
ess_data = {}
for _, row in df_ess.iterrows():
    c = row["country"]
    yr = int(row["year"])
    if c not in ess_data:
        ess_data[c] = {}
    ess_data[c][yr] = {
        "trust": round(float(row["social_trust_score"]), 2),
        "parliament": round(float(row["political_trust_parliament"]), 2),
        "democracy": round(float(row["satisfaction_with_democracy"]), 2),
        "immigrant": round(float(row["immigrant_acceptance_index"]), 2),
    }

ess_coords = {
    "Sweden": [60.1282, 18.6435],
    "Norway": [60.4720, 8.4689],
    "Finland": [61.9241, 25.7482],
    "Germany": [51.1657, 10.4515],
    "France": [46.2276, 2.2137],
    "United Kingdom": [55.3781, -3.4360],
    "Spain": [40.4637, -3.7492],
    "Italy": [41.8719, 12.5674],
    "Poland": [51.9194, 19.1451],
    "Hungary": [47.1625, 19.5033],
    "Greece": [39.0742, 21.8243],
    "Netherlands": [52.1326, 5.2913],
    "Portugal": [39.3999, -8.2245],
    "Ireland": [53.1424, -7.6921],
    "Czechia": [49.8175, 15.4730],
}

# 3. Load V-Dem Global 2024 Snapshot
df_vdem = pd.read_csv(DATA_DIR / "vdem_global_1900_2025.csv")
vdem_latest = df_vdem[df_vdem["year"] == 2024]
if vdem_latest.empty:
    vdem_latest = df_vdem[df_vdem["year"] == df_vdem["year"].max()]

vdem_data = {}
for _, row in vdem_latest.iterrows():
    code = str(row["country_text_id"])
    name = str(row["country_name"])
    vdem_data[code] = {
        "name": name,
        "libdem": round(float(row["v2x_libdem"]), 3) if pd.notna(row["v2x_libdem"]) else 0.0,
        "polyarchy": round(float(row["v2x_polyarchy"]), 3) if pd.notna(row["v2x_polyarchy"]) else 0.0,
        "rule": round(float(row["v2x_rule"]), 3) if pd.notna(row["v2x_rule"]) else 0.0,
        "expression": round(float(row["v2x_freexp_altinf"]), 3) if pd.notna(row["v2x_freexp_altinf"]) else 0.0,
        "clean_gov": round(1.0 - float(row["v2x_corr"]), 3) if pd.notna(row["v2x_corr"]) else 0.0,
        "academ": round(float(row["v2xca_academ"]), 3) if ("v2xca_academ" in row and pd.notna(row["v2xca_academ"])) else 0.0,
    }

# 4. Load Global Inequality Data
df_ineq = pd.read_csv(DATA_DIR / "global_development_inequality.csv")
ineq_data = {}
for _, row in df_ineq.iterrows():
    reg = str(row["region"])
    yr = int(row["year"])
    if reg not in ineq_data:
        ineq_data[reg] = {}
    ineq_data[reg][yr] = {
        "gdp": round(float(row["gdp_per_capita_k"]), 1),
        "gini": round(float(row["gini_inequality_index"]), 1),
        "life_exp": round(float(row["life_expectancy"]), 1),
        "edu": round(float(row["education_index"]), 2),
        "wellbeing": round(float(row["civic_wellbeing_score"]), 1),
        "media_trust": round(float(row["trust_in_media_pct"]), 1),
    }

region_coords = {
    "North America": [45.0, -100.0],
    "Western Europe": [48.0, 5.0],
    "Nordic Countries": [63.0, 15.0],
    "Eastern Europe": [52.0, 30.0],
    "East Asia": [35.0, 115.0],
    "Latin America": [-15.0, -60.0],
    "South Asia": [22.0, 78.0],
    "Sub-Saharan Africa": [0.0, 25.0],
    "Middle East & North Africa": [28.0, 35.0],
}

# Major Global Country Centroids for V-Dem mapping
global_centroids = {
    "USA": [37.0902, -95.7129], "CAN": [56.1304, -106.3468], "MEX": [23.6345, -102.5528], "BRA": [-14.2350, -51.9253],
    "ARG": [-38.4161, -63.6167], "CHL": [-35.6751, -71.5430], "COL": [4.5709, -74.2973], "PER": [-9.1900, -75.0152],
    "GBR": [55.3781, -3.4360], "FRA": [46.2276, 2.2137], "DEU": [51.1657, 10.4515], "ITA": [41.8719, 12.5674],
    "ESP": [40.4637, -3.7492], "PRT": [39.3999, -8.2245], "SWE": [60.1282, 18.6435], "NOR": [60.4720, 8.4689],
    "FIN": [61.9241, 25.7482], "DNK": [56.2639, 9.5018], "NLD": [52.1326, 5.2913], "BEL": [50.5039, 4.4699],
    "CHE": [46.8182, 8.2275], "AUT": [47.5162, 14.5501], "POL": [51.9194, 19.1451], "HUN": [47.1625, 19.5033],
    "GRC": [39.0742, 21.8243], "ROU": [45.9432, 24.9668], "UKR": [48.3794, 31.1656], "RUS": [61.5240, 105.3188],
    "TUR": [38.9637, 35.2433], "ISR": [31.0461, 34.8516], "SAU": [23.8859, 45.0792], "IRN": [32.4279, 53.6880],
    "EGY": [26.8206, 30.8025], "ZAF": [-30.5595, 22.9375], "NGA": [9.0820, 8.6753], "KEN": [-0.0236, 37.9062],
    "GHA": [7.9465, -1.0232], "IND": [20.5937, 78.9629], "PAK": [30.3753, 69.3451], "BGD": [23.6850, 90.3563],
    "IDN": [-0.7893, 113.9213], "JPN": [36.2048, 138.2529], "KOR": [35.9078, 127.7669], "CHN": [35.8617, 104.1954],
    "AUS": [-25.2744, 133.7751], "NZL": [-40.9006, 174.8860], "TWN": [23.6978, 120.9605], "PHL": [12.8797, 121.7740],
    "THA": [15.8700, 100.9925], "VNM": [14.0583, 108.2772], "SGP": [1.3521, 103.8198], "MYS": [4.2105, 101.9758],
}

payload = {
    "us": {"coords": us_coords, "data": us_data},
    "ess": {"coords": ess_coords, "data": ess_data},
    "vdem": {"centroids": global_centroids, "data": vdem_data},
    "inequality": {"coords": region_coords, "data": ineq_data},
}

with open(WWW_DIR / "storymap_data.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print("Generated www/storymap_data.json successfully!")

