# 🚀 RiskFlow - Production MLOps for Credit Risk

**Real-time credit risk scoring with automated model management and LLM-powered insights**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.22+-orange.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red.svg)](https://streamlit.io/)

## 📊 Overview

RiskFlow is a production-ready MLOps pipeline for credit risk modeling that combines traditional quantitative finance with modern AI capabilities. Built for hedge funds, banks, and trading firms requiring real-time credit risk assessment with automated model management.

### 🎯 Key Features

- **Real-time Credit Risk Scoring**: FastAPI endpoints serving PD/LGD predictions in <100ms
- **Automated Model Management**: MLflow-powered versioning, training, and deployment
- **LLM-Powered Insights**: OpenAI integration for intelligent risk commentary
- **Production Monitoring**: Model drift detection and performance tracking
- **Interactive Dashboard**: Streamlit interface for model management and visualization
- **Container-Ready**: Docker setup for seamless deployment

### 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │───▶│  ML Pipeline    │───▶│  Serving Layer  │
│   • Ingestion   │    │   • Training    │    │   • FastAPI     │
│   • Validation  │    │   • Validation  │    │   • Caching     │
│   • Features    │    │   • Registry    │    │   • Monitoring  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LLM Service    │    │  Model Registry │    │   Dashboard     │
│   • Analysis    │    │   • MLflow      │    │   • Streamlit   │
│   • Insights    │    │   • Versioning  │    │   • Metrics     │
│   • Docs        │    │   • A/B Testing │    │   • Management  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- OpenAI API key (for LLM features)

### Installation

1. **Clone and setup**
```bash
git clone <repository-url>
cd riskflow
python -m venv credit-risk-env
source credit-risk-env/bin/activate  # On Windows: credit-risk-env\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations
```

3. **Initialize database and generate demo data**
```bash
python scripts/setup_environment.py
python scripts/generate_demo_data.py
```

4. **Start services**
```bash
# Terminal 1: API Server
python scripts/run_api.py

# Terminal 2: Dashboard
python scripts/run_dashboard.py

# Terminal 3: MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

### 🔍 Demo URLs
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

## 📚 Usage Examples

### API Credit Risk Scoring
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "features": {
        "annual_income": 75000,
        "debt_to_income": 0.35,
        "credit_history_length": 5,
        "loan_amount": 25000
    }
})

result = response.json()
print(f"PD: {result['probability_default']:.3f}")
print(f"Risk Score: {result['risk_score']}")
print(f"LLM Insight: {result['risk_commentary']}")
```

### Model Training and Management
```python
from src.models.model_training import train_model
from src.monitoring.drift_detection import check_model_drift

# Train new model
model_uri = train_model(
    data_path="data/processed/credit_data.csv",
    experiment_name="credit_risk_models"
)

# Check for drift
drift_detected = check_model_drift(
    model_uri=model_uri,
    new_data_path="data/processed/latest_data.csv"
)
```

## 🛠️ Development

### Project Structure
```
riskflow/
├── src/                    # Core application code
│   ├── api/               # FastAPI application
│   ├── models/            # ML models and training
│   ├── data/              # Data processing
│   ├── llm/               # LLM integration
│   └── monitoring/        # Model monitoring
├── dashboard/             # Streamlit dashboard
├── tests/                 # Test suites
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# Performance tests
pytest tests/performance/
```

### Code Quality
```bash
# Format code
black src/ dashboard/ tests/

# Sort imports
isort src/ dashboard/ tests/

# Lint code
flake8 src/ dashboard/ tests/

# Type checking
mypy src/
```

## 🐳 Docker Deployment

### Single Container
```bash
docker build -t riskflow .
docker run -p 8000:8000 -p 8501:8501 riskflow
```

### Multi-Service with Docker Compose
```bash
docker-compose up
```

## 📈 Model Performance

Current model performance on validation data:
- **AUC-ROC**: 0.82
- **Precision**: 0.78
- **Recall**: 0.75
- **F1-Score**: 0.76
- **Response Time**: <100ms (p95)

## 🔬 Technical Highlights

### Credit Risk Modeling
- **PD Models**: Logistic regression and Random Forest for probability of default
- **LGD Models**: Linear regression with econometric features
- **Feature Engineering**: 50+ derived risk indicators
- **Validation Framework**: Walk-forward validation with business logic checks

### MLOps Capabilities
- **Automated Training**: Scheduled retraining with performance monitoring
- **Model Registry**: MLflow-based versioning and staging
- **Drift Detection**: Statistical and ML-based drift monitoring
- **A/B Testing**: Framework for comparing model versions

### LLM Integration
- **Risk Commentary**: Natural language explanations of risk scores
- **Documentation**: Auto-generated model documentation
- **Insights**: Market context and regulatory compliance notes
- **Cost Optimization**: Intelligent caching and prompt optimization

## 🎯 Business Impact

### For Hedge Funds
- **Alpha Generation**: Enhanced credit risk signals for trading strategies
- **Risk Management**: Real-time portfolio credit exposure monitoring
- **Regulatory Compliance**: Automated model documentation and validation

### For Banks
- **Lending Decisions**: Real-time credit approval workflows
- **Regulatory Reporting**: IFRS9/CECL compliance automation
- **Cost Reduction**: 60% reduction in manual model validation time

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Recognition

Built for the evolving needs of quantitative finance in 2025:
- **ESG Integration Ready**: Framework for ESG risk factors
- **Real-time Capabilities**: Built for high-frequency trading environments  
- **Regulatory Compliant**: Designed with Basel III and IFRS9 requirements
- **AI-Native**: LLM integration for next-generation risk insights

---

**Contact**: Built by Ayush Bhattacharya | [LinkedIn](https://linkedin.com/in/ayush-bhattacharya) | [Email](mailto:ayush@example.com)