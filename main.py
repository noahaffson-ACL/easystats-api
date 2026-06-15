from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd

from stats_engine import detect_variables, run_analysis
from docx_export import build_report

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


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    # Si aucune clé n'est configurée (ex: environnement local), on n'impose pas l'auth.
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


# ── ENDPOINTS ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ThèsIA API"}


class DetectRequest(BaseModel):
    data: List[dict]


@app.post("/detect-variables", dependencies=[Depends(verify_api_key)])
def detect_variables_endpoint(body: DetectRequest):
    try:
        df = pd.DataFrame(body.data)
        return detect_variables(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'analyse des variables : {exc}")


class AnalyzeRequest(BaseModel):
    data: List[dict]
    variable_dependante: str
    variables_independantes: List[str] = []
    type_etude: Optional[str] = None
    groupes: Optional[str] = None


@app.post("/analyze", dependencies=[Depends(verify_api_key)])
def analyze(body: AnalyzeRequest):
    try:
        return run_analysis(
            data=body.data,
            var_dep=body.variable_dependante,
            vars_indep=body.variables_independantes,
            type_etude=body.type_etude,
            groupes=body.groupes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'analyse : {exc}")


@app.post("/export-docx", dependencies=[Depends(verify_api_key)])
def export_docx(body: AnalyzeRequest):
    try:
        results = run_analysis(
            data=body.data,
            var_dep=body.variable_dependante,
            vars_indep=body.variables_independantes,
            type_etude=body.type_etude,
            groupes=body.groupes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'analyse : {exc}")

    buffer = build_report(results)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=rapport_analyse.docx"},
    )
