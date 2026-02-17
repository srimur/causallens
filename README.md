# CausalLens

Causal discovery and counterfactual inference platform.
<img width="1919" height="988" alt="image" src="https://github.com/user-attachments/assets/1cb786d4-c146-4c27-9bd2-ac5eb6df1f4a" />

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn python-multipart scipy
```

### 2. Start the Backend API

```bash
cd causallens
uvicorn causallens.api.server:app --reload --host 0.0.0.0 --port 8000
```

API will be running at: `http://localhost:8000`

### 3. Start the Frontend

Open a **new terminal** and run:

```bash
cd causallens/causallens/app/frontend
python -m http.server 3000
```

Frontend will be running at: `http://localhost:3000`

### 4. Open in Browser

Go to: **http://localhost:3000**

---

---

## Usage

1. Pick a dataset (Business, Sensor, ML Pipeline, etc.)
2. Click **"Run Discovery"** to run the PC algorithm
3. Ask causal questions like:
   - "What if ad_spend increases by 20%?"
   - "Does pressure affect flow_rate?"
   - "How does training_time affect model_accuracy?"

---

## API Endpoints

| Endpoint          | Method | Description              |
| ----------------- | ------ | ------------------------ |
| `/data/demo`      | POST   | Load demo dataset        |
| `/data/load`      | POST   | Upload CSV               |
| `/discover`       | POST   | Run PC algorithm         |
| `/query`          | POST   | Ask causal question      |
| `/ate`            | POST   | Compute treatment effect |
| `/counterfactual` | POST   | Compute counterfactual   |

---

## Project Structure

```
causallens/
├── causallens/
│   ├── api/server.py         # FastAPI backend
│   ├── app/frontend/         # Web UI (white background)
│   ├── discovery/            # PC algorithm, Granger
│   ├── inference/            # Counterfactual engine
│   ├── query/                # NLP parser
│   └── data/                 # Demo datasets
└── requirements.txt
```
