"""
causallens.data.demos
======================
Domain-specific demo datasets for non-econometric use cases.

Each function returns (SEMGenerator, description_dict) so the caller
can generate data and also get metadata about what the variables mean.

All datasets use SEMGenerator from generator.py — no new dependencies.

Design note: Each dataset includes at least one collider structure
(X → Z ← Y where X ⊥ Y) so the PC algorithm can orient edges.
Coefficients are tuned to be large relative to noise so that
structure recovery is reliable at α=0.05 with n≥1000.

Available demos
---------------
- sensor_network():   IoT sensor cascade (temperature → pressure → flow)
- ml_pipeline():      ML system monitoring (data drift → model accuracy → alerts)
- software_metrics(): Software engineering (complexity → bugs → deploy failures)
- network_latency():  Distributed system (load → queue depth → latency → errors)
"""

from __future__ import annotations

from typing import Dict, Tuple

from causallens.data.generator import SEMGenerator


def sensor_network(seed: int = 42) -> Tuple[SEMGenerator, Dict]:
    """
    IoT sensor cascade — models a physical process where upstream
    sensor readings causally affect downstream measurements.

    Structure:
        ambient_temp → pipe_temp → pressure → flow_rate → output_volume
        external_vibration → pressure  (collider: pipe_temp → pressure ← vibration)
        external_vibration → flow_rate (second collider)

    Two independent roots (ambient_temp, external_vibration) create
    the collider structures needed for PC edge orientation.
    """
    gen = SEMGenerator(seed=seed)
    gen.add_variable("ambient_temp", noise_std=5.0, noise_mean=25.0)
    gen.add_variable("external_vibration", noise_std=2.0, noise_mean=5.0)
    gen.add_variable("pipe_temp",
                     parents={"ambient_temp": 0.9},
                     noise_std=1.5, noise_mean=60.0)
    # Collider: pipe_temp → pressure ← external_vibration
    gen.add_variable("pressure",
                     parents={"pipe_temp": 1.2, "external_vibration": 3.0},
                     noise_std=2.0, noise_mean=100.0)
    # Collider: pressure → flow_rate ← external_vibration
    gen.add_variable("flow_rate",
                     parents={"pressure": 0.4, "external_vibration": -1.5},
                     noise_std=1.0, noise_mean=20.0)
    gen.add_variable("output_volume",
                     parents={"flow_rate": 5.0},
                     noise_std=3.0, noise_mean=50.0)

    meta = {
        "name": "Sensor Network",
        "domain": "IoT / Industrial",
        "description": "Physical sensor cascade — upstream readings causally affect downstream measurements.",
        "variables": {
            "ambient_temp": "External temperature (°C)",
            "external_vibration": "Vibration from nearby machinery (mm/s)",
            "pipe_temp": "Internal pipe temperature (°C)",
            "pressure": "System pressure (kPa)",
            "flow_rate": "Fluid flow rate (L/min)",
            "output_volume": "Total output per cycle (L)",
        },
        "ground_truth": "ambient_temp → pipe_temp → pressure → flow_rate → output_volume; external_vibration → pressure; external_vibration → flow_rate",
    }
    return gen, meta


def ml_pipeline(seed: int = 42) -> Tuple[SEMGenerator, Dict]:
    """
    ML system monitoring — models how data quality issues propagate
    through a production ML pipeline to cause downstream failures.

    Structure:
        data_drift → feature_quality → model_accuracy → prediction_confidence
        training_staleness → model_accuracy  (collider: features → accuracy ← staleness)
        training_staleness → prediction_confidence  (second collider)
        data_drift → missing_rate
        model_accuracy → alert_rate

    Two independent roots (data_drift, training_staleness) create colliders.
    """
    gen = SEMGenerator(seed=seed)
    gen.add_variable("data_drift", noise_std=8.0, noise_mean=30.0)
    gen.add_variable("training_staleness", noise_std=5.0, noise_mean=15.0)
    gen.add_variable("missing_rate",
                     parents={"data_drift": 0.3},
                     noise_std=2.0, noise_mean=5.0)
    gen.add_variable("feature_quality",
                     parents={"data_drift": -0.8},
                     noise_std=3.0, noise_mean=85.0)
    # Collider: feature_quality → model_accuracy ← training_staleness
    gen.add_variable("model_accuracy",
                     parents={"feature_quality": 0.6, "training_staleness": -0.5},
                     noise_std=2.0, noise_mean=50.0)
    # Collider: model_accuracy → prediction_conf ← training_staleness
    gen.add_variable("prediction_confidence",
                     parents={"model_accuracy": 0.7, "training_staleness": -0.3},
                     noise_std=2.5, noise_mean=60.0)
    gen.add_variable("alert_rate",
                     parents={"model_accuracy": -0.4},
                     noise_std=1.5, noise_mean=20.0)

    meta = {
        "name": "ML Pipeline Monitoring",
        "domain": "MLOps / Data Science",
        "description": "Models how data quality issues propagate through a production ML pipeline.",
        "variables": {
            "data_drift": "Distribution shift score (higher = more drift)",
            "training_staleness": "Days since last model retrain",
            "missing_rate": "% missing values in incoming data",
            "feature_quality": "Feature pipeline health (0-100)",
            "model_accuracy": "Model performance score",
            "prediction_confidence": "Average prediction probability",
            "alert_rate": "Monitoring alerts per hour",
        },
        "ground_truth": "data_drift → feature_quality → model_accuracy → prediction_confidence; data_drift → missing_rate; training_staleness → model_accuracy; training_staleness → prediction_confidence; model_accuracy → alert_rate",
    }
    return gen, meta


def software_metrics(seed: int = 42) -> Tuple[SEMGenerator, Dict]:
    """
    Software engineering metrics — models how code-level decisions
    propagate to production reliability.

    Structure:
        team_size → code_complexity → bug_count → deploy_failures → incident_count
        test_coverage → bug_count  (collider: complexity → bugs ← test_coverage)
        test_coverage → deploy_failures  (collider: bugs → deploys ← test_coverage)
        code_complexity → review_time

    Two independent roots (team_size, test_coverage) create colliders.
    """
    gen = SEMGenerator(seed=seed)
    gen.add_variable("team_size", noise_std=2.0, noise_mean=8.0)
    gen.add_variable("test_coverage", noise_std=8.0, noise_mean=65.0)
    gen.add_variable("code_complexity",
                     parents={"team_size": 2.0},
                     noise_std=2.0, noise_mean=10.0)
    gen.add_variable("review_time",
                     parents={"code_complexity": 0.5},
                     noise_std=0.8, noise_mean=2.0)
    # Collider: code_complexity → bug_count ← test_coverage
    gen.add_variable("bug_count",
                     parents={"code_complexity": 1.0, "test_coverage": -0.15},
                     noise_std=1.5, noise_mean=5.0)
    # Collider: bug_count → deploy_failures ← test_coverage
    gen.add_variable("deploy_failures",
                     parents={"bug_count": 0.6, "test_coverage": -0.05},
                     noise_std=0.8, noise_mean=2.0)
    gen.add_variable("incident_count",
                     parents={"deploy_failures": 1.5},
                     noise_std=1.0, noise_mean=3.0)

    meta = {
        "name": "Software Engineering Metrics",
        "domain": "DevOps / SRE",
        "description": "Models how code-level decisions propagate to production reliability.",
        "variables": {
            "team_size": "Number of contributors",
            "test_coverage": "% of codebase with automated tests",
            "code_complexity": "Cyclomatic complexity (avg per module)",
            "review_time": "Code review turnaround (hours)",
            "bug_count": "Bugs found per sprint",
            "deploy_failures": "Failed deployments per month",
            "incident_count": "Production incidents per month",
        },
        "ground_truth": "team_size → code_complexity → bug_count → deploy_failures → incident_count; test_coverage → bug_count; test_coverage → deploy_failures; code_complexity → review_time",
    }
    return gen, meta


def network_latency(seed: int = 42) -> Tuple[SEMGenerator, Dict]:
    """
    Distributed system performance — models how load propagates
    through a service to cause user-visible latency.

    Structure:
        request_load → queue_depth → processing_time → response_latency → error_rate
        gc_pressure → processing_time  (collider: queue → processing ← gc)
        gc_pressure → error_rate       (second collider)

    Two independent roots (request_load, gc_pressure) create colliders.
    """
    gen = SEMGenerator(seed=seed)
    gen.add_variable("request_load", noise_std=100.0, noise_mean=1000.0)
    gen.add_variable("gc_pressure", noise_std=5.0, noise_mean=20.0)
    gen.add_variable("queue_depth",
                     parents={"request_load": 0.03},
                     noise_std=2.0, noise_mean=10.0)
    # Collider: queue_depth → processing_time ← gc_pressure
    gen.add_variable("processing_time",
                     parents={"queue_depth": 3.0, "gc_pressure": 2.0},
                     noise_std=5.0, noise_mean=50.0)
    gen.add_variable("response_latency",
                     parents={"processing_time": 1.8},
                     noise_std=8.0, noise_mean=100.0)
    # Collider: response_latency → error_rate ← gc_pressure
    gen.add_variable("error_rate",
                     parents={"response_latency": 0.03, "gc_pressure": 0.15},
                     noise_std=0.5, noise_mean=1.0)

    meta = {
        "name": "Network Latency",
        "domain": "Distributed Systems",
        "description": "Models how request load propagates through a service to cause latency and errors.",
        "variables": {
            "request_load": "Incoming requests per second",
            "gc_pressure": "Garbage collection pressure (%)",
            "queue_depth": "Requests waiting in queue",
            "processing_time": "Time to process one request (ms)",
            "response_latency": "End-to-end response time (ms)",
            "error_rate": "Error responses (%)",
        },
        "ground_truth": "request_load → queue_depth → processing_time → response_latency → error_rate; gc_pressure → processing_time; gc_pressure → error_rate",
    }
    return gen, meta
