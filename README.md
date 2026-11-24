# 🛡️ FinShield - Autonomous Banking Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An AI/ML-based autonomous banking fraud detection system for detecting suspicious mule accounts using a 6-layer architecture.**

## 🏗️ Architecture Overview

FinShield implements a **6-layer architecture** for comprehensive fraud detection:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 6: INTERFACE                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Streamlit      │  │  Real-time      │  │  Agent Chat         │ │
│  │  Dashboard      │  │  Velocity       │  │  Interface          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 5: AUTONOMOUS RESPONSE                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Ollama/Llama3  │  │  LangGraph      │  │  RAG + ChromaDB     │ │
│  │  LLM Reasoning  │  │  State Machine  │  │  Knowledge Base     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│  Actions: MONITOR → INVESTIGATE → REQUEST_INFO → FREEZE → SAR → ESCALATE → CLOSE
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 4: RISK ENGINE                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Composite      │  │  Watchlist      │  │  Cross-Channel      │ │
│  │  Scoring        │  │  Manager        │  │  Identity Resolution│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│  Weights: ML(40%) + Graph(30%) + Regulatory(20%) + Anomaly(10%)
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: GRAPH INTELLIGENCE                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Cycle          │  │  Kingpin        │  │  Shared Entity      │ │
│  │  Detection      │  │  Detection      │  │  Detection          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│  NetworkX • PageRank • Betweenness • Clique Detection
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: AI/ML DETECTION                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Anomaly        │  │  Supervised     │  │  Explainability     │ │
│  │  Detection      │  │  Classification │  │  (SHAP)             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│  PyOD Ensemble • XGBoost + SMOTE • SHAP Values
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: DATA INGESTION                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  FastAPI REST   │  │  DuckDB         │  │  Synthetic Data     │ │
│  │  Endpoints      │  │  Storage        │  │  Generator          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│  Kafka Streaming • FMS/TMS Alerts • Government Watchlists
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

### Fraud Patterns Detected
- **Structuring/Smurfing**: Multiple small transactions to evade reporting thresholds
- **Velocity Abuse**: Unusually high transaction frequency
- **Dormancy Breaks**: Sudden activity after long inactivity
- **Circular Flows**: Money cycling through account networks
- **Mule Rings**: Clusters of interconnected mule accounts
- **Kingpin Accounts**: Aggregator accounts receiving from many mules
- **Shared Devices/IPs**: Multiple accounts using same device or network

### Technical Capabilities
- **Offline/Air-gapped Operation**: Fully functional without internet (Ollama local LLM)
- **Real-time Streaming**: Kafka-based transaction ingestion
- **Explainable AI**: SHAP values for every classification decision
- **Autonomous Response**: LLM-driven investigation and case management
- **Regulatory Compliance**: SAR generation, watchlist screening, audit trails
- **Scalable Architecture**: Modular layers, Docker-ready deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for full stack)
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for Ollama acceleration)

### Option 1: Docker Compose (Recommended)

```bash
# Clone and navigate
git clone https://github.com/yourusername/finshield.git
cd finshield

# Start all services
docker-compose up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# ChromaDB: http://localhost:8001
# Kafka: localhost:9092
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data & train models
python -m src.detection.classifier

# Start API server
uvicorn src.ingestion.api:app --reload --port 8000

# Start dashboard (in another terminal)
streamlit run src/interface/dashboard.py
```

### Option 3: Minimal Demo

```bash
# Quick demo without Docker
python -c "
from src.ingestion.synthetic_data import generate_all_data
from src.detection.classifier import train_full_pipeline
from src.agent.response import run_agent_demo

# Generate data
data = generate_all_data(n_legitimate=100, n_mule=20, n_suspicious=10, n_kingpin=3)

# Train models
train_full_pipeline()

# Run agent demo
run_agent_demo()
"
```

## 📁 Project Structure

```
finshield/
├── config.yaml                 # Central configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Multi-service orchestration
├── README.md                   # This file
├── LICENSE                     # MIT License
├── data/                       # Data directory (gitignored)
│   ├── chromadb/              # Vector database
│   ├── ollama/                # LLM models
│   └── duckdb/                # Database files
├── models/                     # Trained models (gitignored)
│   ├── anomaly/               # Anomaly detector models
│   └── classifier/            # XGBoost classifier
├── logs/                       # Application logs
├── monitoring/                 # Prometheus/Grafana configs
├── src/
│   ├── common/                 # Shared utilities
│   │   ├── config.py          # Configuration loader
│   │   └── models.py          # Pydantic data models
│   ├── ingestion/              # Layer 1: Data Ingestion
│   │   ├── api.py             # FastAPI REST endpoints
│   │   ├── database.py        # DuckDB manager
│   │   └── synthetic_data.py  # Test data generator
│   ├── detection/              # Layer 2: AI/ML Detection
│   │   ├── features.py        # Feature engineering (tsfresh)
│   │   ├── anomaly.py         # PyOD ensemble detection
│   │   └── classifier.py      # XGBoost + SHAP
│   ├── graph/                  # Layer 3: Graph Intelligence
│   │   └── analysis.py        # NetworkX analysis
│   ├── risk/                   # Layer 4: Risk Engine
│   │   └── engine.py          # Composite scoring + watchlist
│   ├── agent/                  # Layer 5: Autonomous Response
│   │   └── response.py        # LangGraph + Ollama + RAG
│   └── interface/              # Layer 6: Interface
│       └── dashboard.py       # Streamlit dashboard
└── tests/                      # Unit & integration tests
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Key sections:
ingestion:
  api:
    host: "0.0.0.0"
    port: 8000
  database:
    path: "data/duckdb/finshield.duckdb"
  kafka:
    bootstrap_servers: "localhost:9092"

detection:
  anomaly:
    contamination: 0.05
    n_estimators: 100
  xgboost:
    n_estimators: 200
    max_depth: 6
  smote:
    k_neighbors: 5

graph:
  networkx:
    min_cycle_length: 3
    max_cycle_length: 10
  kingpin:
    min_in_degree: 5

risk_engine:
  weights:
    ml: 0.4
    graph: 0.3
    regulatory: 0.2
    anomaly: 0.1
  thresholds:
    watch: 30
    alert: 60
    act: 80

agent:
  ollama:
    base_url: "http://localhost:11434"
    model: "llama3"
  rag:
    chroma_host: "localhost"
    chroma_port: 8000
```

## 🔌 API Endpoints

### Transaction Ingestion
```bash
# Single transaction
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC123",
    "amount": 50000,
    "channel": "IMPS",
    "txn_type": "CREDIT",
    "beneficiary_account_id": "ACC456"
  }'

# Batch transactions
curl -X POST http://localhost:8000/transactions/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

### Alerts & Watchlist
```bash
# FMS Alert
curl -X POST http://localhost:8000/alerts/fms \
  -H "Content-Type: application/json" \
  -d '{"alert_id": "FMS001", "account_id": "ACC123", "alert_type": "HIGH_VALUE", "severity": "HIGH"}'

# TMS Alert
curl -X POST http://localhost:8000/alerts/tms \
  -H "Content-Type: application/json" \
  -d '{"alert_id": "TMS001", "account_id": "ACC123", "alert_type": "VELOCITY", "severity": "MEDIUM"}'

# Watchlist
curl -X POST http://localhost:8000/watchlist \
  -H "Content-Type: application/json" \
  -d '{"source": "UN_SANCTIONS", "entity_name": "John Doe", "severity": "CRITICAL"}'
```

### Queries
```bash
# Account transactions
curl http://localhost:8000/accounts/ACC123/transactions?days=30

# Account risk profile
curl http://localhost:8000/accounts/ACC123/risk

# Health check
curl http://localhost:8000/health
```

## 🤖 Autonomous Agent Actions

The agent operates on a **7-node LangGraph state machine**:

| Action | Trigger | Description |
|--------|---------|-------------|
| **MONITOR** | Low risk | Continue surveillance, log for review |
| **INVESTIGATE** | Medium risk | Deep dive: graph, watchlist, patterns |
| **REQUEST_INFO** | Missing data | Ask for KYC, source of funds, docs |
| **FREEZE_ACCOUNT** | High risk + confirmation | Simulate core banking freeze API |
| **GENERATE_SAR** | Confirmed suspicious | Create Suspicious Activity Report |
| **ESCALATE** | Complex/high-value | Route to senior analyst/compliance |
| **CLOSE_CASE** | False positive | Document resolution, feedback loop |

## 📊 Dashboard Pages

1. **Overview** - Portfolio risk summary, tier distribution, top risk accounts
2. **Account Explorer** - Deep-dive: ML scores, SHAP, graph metrics, transactions
3. **Network Graph** - Interactive pyvis visualization of mule networks
4. **Agent Actions** - Run autonomous agent on selected accounts
5. **Velocity Monitor** - Real-time transaction velocity charts
6. **Risk Engine** - Configure weights, thresholds, test scoring
7. **Watchlist** - Manage government watchlists, cross-channel resolution
8. **Cases & SARs** - Case management, SAR report tracking
9. **Model Training** - Full pipeline, performance metrics, feature importance
10. **Settings** - System config, data generation, health checks

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Test specific module
pytest tests/test_detection.py -v
```

## 📈 Model Performance

Expected performance on synthetic data:
- **Accuracy**: ~92%
- **MULE Recall**: ~89%
- **SUSPICIOUS Recall**: ~85%
- **LEGITIMATE Precision**: ~95%
- **ROC AUC (MULE)**: ~0.94

## 🛡️ Security & Compliance

- **Data Encryption**: At-rest (DuckDB) and in-transit (TLS)
- **Audit Logging**: All agent actions, decisions, API calls logged
- **Access Control**: Role-based (Analyst, Senior, Compliance, Admin)
- **Regulatory**: SAR auto-generation, watchlist screening, STR filing
- **Air-gapped**: Fully operational offline with local LLM

## 🔧 Extending FinShield

### Adding New Fraud Patterns
1. Add feature engineering in `src/detection/features.py`
2. Update synthetic data generator in `src/ingestion/synthetic_data.py`
3. Retrain models: `python -m src.detection.classifier`

### Custom Watchlist Sources
```python
from src.risk.engine import WatchlistManager

wm = WatchlistManager()
wm.load_from_file("CUSTOM_SOURCE", "watchlist.json")
```

### New Agent Actions
Extend `src/agent/response.py`:
1. Add node to `AgentState`
2. Implement action function
3. Update routing logic in `should_continue()`

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🙏 Acknowledgments

- **PyOD** for anomaly detection algorithms
- **XGBoost** for gradient boosting
- **SHAP** for explainability
- **NetworkX** for graph analytics
- **LangGraph** for agent orchestration
- **Ollama** for local LLM inference
- **ChromaDB** for vector storage
- **Streamlit** for rapid dashboard development

---

<div align="center">
  <strong>Built for demonstrating production-grade fraud detection capabilities</strong>
  <br>
  <em>Ready for GitHub showcase and resume portfolio</em>
</div>