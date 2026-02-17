"""
causallens.api.server
=======================
REST API for CausalLens using FastAPI.

Exposes the full pipeline as HTTP endpoints:
  POST /data/load          — load a CSV dataset
  POST /data/demo          — load a built-in demo dataset
  GET  /data/info          — get current dataset info
  POST /discover           — run PC algorithm
  GET  /graph              — get discovered graph
  GET  /graph/edges        — list edges
  GET  /graph/path         — find causal path between two variables
  POST /query              — ask a natural language causal question
  POST /query/explain      — ask + get full explainability breakdown
  POST /ate                — compute average treatment effect
  POST /counterfactual     — compute a specific counterfactual

Run:
    uvicorn causallens.api.server:app --reload
    # or
    python -m causallens.api.server

Dependencies:
    pip install fastapi uvicorn python-multipart
"""

from __future__ import annotations

import io
import sys
import os
import time
from typing import Any, Dict, List, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from causallens.types import CausalGraph, EdgeType
from causallens.discovery.pc import PCAlgorithm
from causallens.inference.counterfactual import CounterfactualEngine
from causallens.query.parser import QueryParser, QueryType, InterventionType
from causallens.query.responder import QueryResponder
from causallens.query.explainer import QueryExplainer
from causallens.data.generator import SEMGenerator
from causallens.data.demos import sensor_network, ml_pipeline, software_metrics, network_latency
from causallens.graph.visualize import find_causal_path

# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CausalLens API",
    description="REST API for causal discovery, counterfactual inference, and natural language causal queries.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.graph: Optional[CausalGraph] = None
        self.engine: Optional[CounterfactualEngine] = None
        self.parser: Optional[QueryParser] = None
        self.explainer: Optional[QueryExplainer] = None
        self.data_name: str = ""

state = AppState()

# ── Request/Response Models ──────────────────────────────────────────

class DemoRequest(BaseModel):
    name: str  # "business", "chain", "diamond", "sensor", "ml_pipeline", "software", "network"
    n_samples: int = 2000

class DiscoverRequest(BaseModel):
    alpha: float = 0.05

class QueryRequest(BaseModel):
    question: str

class ATERequest(BaseModel):
    treatment: str
    outcome: str
    treatment_value: float = 1.0
    control_value: float = 0.0
    max_samples: int = 500

class CounterfactualRequest(BaseModel):
    interventions: Dict[str, float]  # {variable: value}
    target: str

class PathRequest(BaseModel):
    source: str
    target: str

# ── Helpers ──────────────────────────────────────────────────────────

def _graph_to_dict(graph: CausalGraph) -> Dict[str, Any]:
    return {
        "variables": graph.variables,
        "n_variables": graph.n_variables,
        "n_edges": graph.n_edges,
        "n_directed": graph.n_directed,
        "n_undirected": graph.n_undirected,
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "type": e.edge_type.value,
                "p_value": e.p_value,
                "weight": e.weight,
            }
            for e in graph.edges
        ],
        "metadata": graph.metadata,
    }

def _require_data():
    if state.data is None:
        raise HTTPException(400, "No data loaded. POST /data/load or /data/demo first.")

def _require_graph():
    _require_data()
    if state.graph is None:
        raise HTTPException(400, "No graph discovered. POST /discover first.")

def _require_engine():
    _require_graph()
    if state.engine is None:
        raise HTTPException(400, "No counterfactual engine. Run discovery first and ensure directed edges exist.")

def _fit_engine():
    """Fit engine, parser, and explainer after discovery."""
    if state.graph.n_directed > 0:
        engine = CounterfactualEngine(model_type="linear")
        engine.fit(state.data, state.graph)
        state.engine = engine
        state.parser = QueryParser.from_graph(state.graph)

        # Build coefficient map for explainer
        coef_map = {}
        for var in state.graph.variables:
            c = engine.get_coefficients(var)
            if c:
                coef_map[var] = c
        state.explainer = QueryExplainer(state.graph, coef_map)

# ── Endpoints: Data ──────────────────────────────────────────────────

@app.post("/data/load")
async def load_csv(file: UploadFile = File(...)):
    """Upload a CSV file as the working dataset."""
    content = await file.read()
    raw = pd.read_csv(io.BytesIO(content))
    numeric = raw.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[1] < 2:
        raise HTTPException(400, f"Need ≥2 numeric columns, got {numeric.shape[1]}.")

    state.data = numeric
    state.graph = None
    state.engine = None
    state.parser = None
    state.data_name = file.filename

    return {
        "status": "ok",
        "filename": file.filename,
        "rows": numeric.shape[0],
        "columns": numeric.shape[1],
        "variables": list(numeric.columns),
    }


DEMO_GENERATORS = {
    "business": lambda s: (SEMGenerator.full_demo(seed=s), "Business Demo"),
    "chain":    lambda s: (SEMGenerator.chain(n_vars=4, seed=s), "Chain (A→B→C→D)"),
    "diamond":  lambda s: (SEMGenerator.diamond(seed=s), "Diamond"),
    "sensor":   lambda s: (sensor_network(seed=s)[0], "Sensor Network"),
    "ml_pipeline": lambda s: (ml_pipeline(seed=s)[0], "ML Pipeline"),
    "software": lambda s: (software_metrics(seed=s)[0], "Software Metrics"),
    "network":  lambda s: (network_latency(seed=s)[0], "Network Latency"),
}

@app.post("/data/demo")
async def load_demo(req: DemoRequest):
    """Load a built-in demo dataset."""
    if req.name not in DEMO_GENERATORS:
        raise HTTPException(400, f"Unknown demo: {req.name}. Options: {list(DEMO_GENERATORS.keys())}")

    gen, label = DEMO_GENERATORS[req.name](42)
    data, _ = gen.generate(n=req.n_samples)

    state.data = data
    state.graph = None
    state.engine = None
    state.parser = None
    state.data_name = label

    return {
        "status": "ok",
        "name": label,
        "rows": data.shape[0],
        "columns": data.shape[1],
        "variables": list(data.columns),
    }


@app.get("/data/info")
async def data_info():
    """Get info about the currently loaded dataset."""
    _require_data()
    return {
        "name": state.data_name,
        "rows": state.data.shape[0],
        "columns": state.data.shape[1],
        "variables": list(state.data.columns),
        "summary": state.data.describe().to_dict(),
    }


@app.get("/data/demos")
async def list_demos():
    """List available demo datasets."""
    return {"demos": list(DEMO_GENERATORS.keys())}


# ── Endpoints: Discovery ─────────────────────────────────────────────

@app.post("/discover")
async def discover(req: DiscoverRequest):
    """Run the PC algorithm on the loaded data."""
    _require_data()
    pc = PCAlgorithm(alpha=req.alpha)
    graph = pc.fit(state.data)
    state.graph = graph
    _fit_engine()

    return {
        "status": "ok",
        "graph": _graph_to_dict(graph),
        "has_engine": state.engine is not None,
    }


# ── Endpoints: Graph ─────────────────────────────────────────────────

@app.get("/graph")
async def get_graph():
    """Get the discovered causal graph."""
    _require_graph()
    return _graph_to_dict(state.graph)


@app.get("/graph/edges")
async def get_edges():
    """List all discovered edges."""
    _require_graph()
    return {
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "type": e.edge_type.value,
                "p_value": e.p_value,
            }
            for e in state.graph.edges
        ]
    }


@app.post("/graph/path")
async def get_path(req: PathRequest):
    """Find a directed causal path between two variables."""
    _require_graph()
    path = find_causal_path(state.graph, req.source, req.target)
    return {
        "source": req.source,
        "target": req.target,
        "path": path,
        "exists": path is not None,
        "length": len(path) if path else 0,
    }


# ── Endpoints: Query ─────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
    """Ask a natural language causal question."""
    _require_engine()
    try:
        parsed = state.parser.parse(req.question)
    except ValueError as e:
        raise HTTPException(400, f"Couldn't parse query: {e}")

    responder = QueryResponder()
    result = _execute_query(parsed)
    answer = responder.respond(parsed, result)

    return {
        "question": req.question,
        "answer": answer,
        "query_type": parsed.query_type.value,
        "result": _sanitize(result),
    }


@app.post("/query/explain")
async def query_explain(req: QueryRequest):
    """Ask a question and get full explainability breakdown."""
    _require_engine()
    try:
        parsed = state.parser.parse(req.question)
    except ValueError as e:
        raise HTTPException(400, f"Couldn't parse query: {e}")

    responder = QueryResponder()
    result = _execute_query(parsed)
    answer = responder.respond(parsed, result)

    parse_explanation = state.explainer.explain_parse(parsed)
    causal_explanation = state.explainer.explain_result(parsed, result)

    return {
        "question": req.question,
        "answer": answer,
        "query_type": parsed.query_type.value,
        "result": _sanitize(result),
        "parse_explanation": parse_explanation.to_dict(),
        "causal_explanation": causal_explanation.to_dict(),
    }


# ── Endpoints: Inference ─────────────────────────────────────────────

@app.post("/ate")
async def compute_ate(req: ATERequest):
    """Compute Average Treatment Effect."""
    _require_engine()
    n = min(req.max_samples, len(state.data))
    result = state.engine.ate(
        state.data.head(n),
        treatment=req.treatment,
        target=req.outcome,
        treatment_value=req.treatment_value,
        control_value=req.control_value,
    )
    return _sanitize(result)


@app.post("/counterfactual")
async def compute_counterfactual(req: CounterfactualRequest):
    """Compute a specific counterfactual."""
    _require_engine()
    obs = state.data.mean()
    result = state.engine.counterfactual(obs, req.interventions, req.target)
    return _sanitize(result)


# ── Internal helpers ─────────────────────────────────────────────────

def _execute_query(parsed) -> Dict[str, Any]:
    """Execute a parsed query against the engine."""
    if parsed.query_type == QueryType.COUNTERFACTUAL:
        obs = state.data.mean()
        iv = parsed.intervention_var
        val = parsed.intervention_value
        t = parsed.intervention_type
        final = {
            InterventionType.SET: val,
            InterventionType.INCREASE_BY: float(obs[iv]) + val,
            InterventionType.DECREASE_BY: float(obs[iv]) - val,
            InterventionType.MULTIPLY_BY: float(obs[iv]) * val,
            InterventionType.INCREASE_PCT: float(obs[iv]) * (1 + val / 100),
            InterventionType.DECREASE_PCT: float(obs[iv]) * (1 - val / 100),
        }.get(t, val)
        return state.engine.counterfactual(obs, {iv: final}, parsed.target)

    elif parsed.query_type == QueryType.ATE:
        return state.engine.ate(
            state.data.head(500),
            treatment=parsed.intervention_var,
            target=parsed.target,
        )

    elif parsed.query_type == QueryType.CAUSAL_CHECK:
        path = find_causal_path(state.graph, parsed.intervention_var, parsed.target)
        is_direct = state.graph.has_edge(parsed.intervention_var, parsed.target, directed_only=True)
        return {
            "has_causal_path": path is not None,
            "is_direct_cause": is_direct,
            "causal_path": path,
        }

    return {}


def _sanitize(d: Dict) -> Dict:
    """Convert numpy types to native Python for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _sanitize(v)
        else:
            out[k] = v
    return out


# ── Run directly ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
