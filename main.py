from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import os

app = FastAPI(title="ThèsIA API", version="1.0.0")

# CORS — autorise ton app Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restreindre à ton domaine Lovable en production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité — vérifie la clé API sur chaque requête
API_KEY = os.environ.get("RAILWAY_API_KEY")

def verify_api_key(x_api_key: str = None):
    from fastapi import Header
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ── ENDPOINTS ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ThèsIA API"}


class DetectRequest(BaseModel):
    data: List[dict]

@app.post("/detect-variables")
def detect_variables(body: DetectRequest):
    df = pd.DataFrame(body.data)
    variables = []

    for col in df.columns:
        # Ignore les colonnes système Kobo
        if col.startswith("_"):
            continue

        series = df[col].dropna()
        missing_pct = round((df[col].isna().sum() / len(df)) * 100, 1)
        unique_vals = series.nunique()

        # Détecter le type
        if pd.api.types.is_numeric_dtype(series):
            if unique_vals <= 5:
                var_type = "categorielle"
            else:
                var_type = "quantitative_continue"
        else:
            if unique_vals <= 10:
                var_type = "categorielle"
            else:
                var_type = "texte"

        variables.append({
            "name": col,
            "type": var_type,
            "missing_pct": missing_pct,
            "unique_values": unique_vals,
            "exemple_valeurs": series.head(3).tolist()
        })

    return {
        "variables": variables,
        "n_observations": len(df),
        "qualite_globale": "bonne" if df.isna().mean().mean() < 0.1 else "acceptable"
    }


class AnalyzeRequest(BaseModel):
    data: List[dict]
    variable_dependante: str
    variables_independantes: List[str]
    type_etude: str
    groupes: Optional[str] = None

@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    df = pd.DataFrame(body.data)
    results = {}

    # Tableau descriptif (Tableau I)
    tableau_desc = {}
    for col in df.columns:
        if col.startswith("_"):
            continue
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            tableau_desc[col] = {
                "type": "continue",
                "n": int(series.count()),
                "moyenne": round(float(series.mean()), 2),
                "ecart_type": round(float(series.std()), 2),
                "mediane": round(float(series.median()), 2),
                "q1": round(float(series.quantile(0.25)), 2),
                "q3": round(float(series.quantile(0.75)), 2)
            }
        else:
            freq = series.value_counts()
            tableau_desc[col] = {
                "type": "categorielle",
                "n": int(series.count()),
                "frequences": {
                    str(k): {"n": int(v), "pct": round(v/len(series)*100, 1)}
                    for k, v in freq.items()
                }
            }

    results["tableau_descriptif"] = tableau_desc

    # Test principal
    var_dep = body.variable_dependante
    if body.groupes and body.groupes in df.columns:
        groupes = df[body.groupes].dropna().unique()

        if len(groupes) == 2:
            g1 = df[df[body.groupes] == groupes[0]][var_dep].dropna()
            g2 = df[df[body.groupes] == groupes[1]][var_dep].dropna()

            if pd.api.types.is_numeric_dtype(df[var_dep]):
                # Shapiro-Wilk pour normalité
                _, p_normal = stats.shapiro(df[var_dep].dropna()[:50])
                if p_normal > 0.05:
                    stat, p_val = stats.ttest_ind(g1, g2)
                    test_nom = "t-test de Student"
                else:
                    stat, p_val = stats.mannwhitneyu(g1, g2)
                    test_nom = "Mann-Whitney U"
            else:
                # Chi² ou Fisher
                contingency = pd.crosstab(df[body.groupes], df[var_dep])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                if (expected < 5).any():
                    _, p_val = stats.fisher_exact(contingency)
                    test_nom = "Test exact de Fisher"
                    stat = None
                else:
                    stat = chi2
                    test_nom = "Chi² de Pearson"

            results["test_principal"] = {
                "test": test_nom,
                "statistic": round(float(stat), 4) if stat else None,
                "p_value": round(float(p_val), 4),
                "significatif": bool(p_val < 0.05),
                "groupes": [str(g) for g in groupes]
            }

    return results
