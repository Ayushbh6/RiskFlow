# 🚀 RiskFlow - Production MLOps for Credit Risk

**Real-time credit risk scoring with automated model management and LLM-powered insights**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.22+-orange.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red.svg)](https://streamlit.io/)

## 📊 Overview

RiskFlow is a production-ready MLOps pipeline for credit risk modeling that combines traditional quantitative finance with modern AI capabilities. Built for hedge funds, banks, and trading firms requiring real-time credit risk assessment with automated model management.

### 🎯 Key Features

- **Real-time Credit Risk Scoring**: FastAPI endpoints serving PD/LGD predictions in <100ms
- **Automated Model Management**: MLflow-powered versioning, training, and deployment
- **LLM-Powered Insights**: OpenAI/Ollama integration for intelligent risk commentary
- **Production Monitoring**: Model drift detection and performance tracking
- **Interactive Dashboard**: Streamlit interface for model management and visualization
- **100% Free Option**: Can run entirely with free tools (Ollama + free APIs)

---

## 🚀 SUPER EASY START - Just Run One Command!

**For complete beginners - this will set up everything automatically:**

```bash
# Download or clone the project first, then:
./start-app.sh
```

**That's it!** The script will:
- ✅ Check all requirements
- ✅ Install Python dependencies
- ✅ Set up database
- ✅ Start both API and dashboard
- ✅ Open everything in your browser

**After running, you'll have:**
- 📊 **Dashboard**: http://localhost:8501
- 🔧 **API**: http://localhost:8000  
- 📖 **API Docs**: http://localhost:8000/docs

---

## 📋 Prerequisites (One-Time Setup)

### Required (Free):
- **Python 3.8+** - [Download here](https://www.python.org/downloads/)
- **Git** - [Download here](https://git-scm.com/downloads)

### Optional (For AI Features):
- **OpenAI API Key** - [Get free $5 credit](https://platform.openai.com/signup)
- **Ollama** - [Free local AI](https://ollama.ai/download) (Alternative to OpenAI)

### For Real Financial Data (Optional but Recommended):
- **FRED API Key** - [Free registration](https://fred.stlouisfed.org/docs/api/api_key.html) (Federal Reserve Economic Data)
- **Tavily API Key** - [Paid service](https://tavily.com) ($5-10/month for credit spread data)

---

## 🎯 Step-by-Step Guide for Absolute Beginners

### Step 1: Download the Project
```bash
# Option A: If you have git
git clone https://github.com/yourusername/RiskFlow.git
cd RiskFlow

# Option B: Download ZIP from GitHub and extract it
# Then navigate to the folder in terminal/command prompt
```

### Step 2: Run the Magic Script
```bash
# On Mac/Linux:
./start-app.sh

# On Windows:
bash start-app.sh
```

### Step 3: Open Your Browser
The script will automatically tell you where to go, but typically:
- **Main Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

### Step 4: Set Up API Keys (Optional)
1. **Create `.env` file** (the script creates a template for you)
2. **Add your API keys**:
   ```bash
   # Edit the .env file that was created
   
   # For AI features (choose one):
   OPENAI_API_KEY=your_openai_key_here
   # OR use free Ollama (see below)
   
   # For real financial data (optional but recommended):
   FRED_API_KEY=your_fred_key_here        # Free from fred.stlouisfed.org
   TAVILY_API_KEY=your_tavily_key_here    # Paid service for credit spreads
   ```

**⚠️ Important Data Sources:**
- **Without API keys**: App runs with limited demo data
- **With FRED key**: Real economic data (unemployment, fed rates) 
- **With Tavily key**: Current credit spread data ($5-10/month)

---

## 🆓 100% Free Setup Options

### Option 1: Basic Setup (Completely Free)
- Just run `./start-app.sh`
- Uses demo economic data (no real market data)
- No AI risk commentary
- Perfect for learning and testing the MLOps pipeline

### Option 2: Free Local AI with Ollama
```bash
# Install Ollama (free)
curl -fsSL https://ollama.ai/install.sh | sh

# Download a free model
ollama pull llama2

# Edit .env file:
LLM_PROVIDER=ollama
```

### Option 3: Real Data + Free AI (Recommended for Testing)
1. Get free FRED API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Install Ollama for free AI
3. Add to `.env` file:
   ```
   FRED_API_KEY=your_fred_key_here
   LLM_PROVIDER=ollama
   ```

### Option 4: Full Production Setup (Best Experience)
1. Get FRED API key (free) + OpenAI credits ($5)
2. Optional: Tavily API for credit spreads ($5-10/month)
3. Add to `.env` file:
   ```
   FRED_API_KEY=your_fred_key_here
   OPENAI_API_KEY=sk-your-key-here
   TAVILY_API_KEY=your_tavily_key_here  # Optional
   LLM_PROVIDER=openai
   ```

---

## 🎮 How to Use the App

### Dashboard Features:

#### 1. **🏠 Overview Tab**
- See real system health
- Monitor API performance  
- View prediction statistics
- Check resource usage (CPU, memory, disk)

#### 2. **📈 Model Performance Tab**
- View trained models
- See accuracy metrics
- Compare model versions
- Monitor model drift

#### 3. **🔄 Real-time Predictions Tab**
- **Interactive Risk Calculator**:
  - Enter loan amount, income, credit score
  - Get instant PD (Probability of Default) score
  - See risk classification (Low/Medium/High)
  - Get AI-powered risk explanation

#### 4. **⚠️ Alerts & Monitoring Tab**
- System health monitoring
- Performance alerts
- Resource usage charts
- Error tracking

#### 5. **🗄️ Model Registry Tab**
- Browse all trained models
- Model versioning
- Performance history
- Model deployment status

### API Features:

#### Making Predictions:
```bash
# Test the API with curl:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "annual_income": 75000,
       "debt_to_income": 0.35,
       "credit_score": 720,
       "loan_amount": 25000
     }'
```

#### Batch Predictions:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "predictions": [
         {"annual_income": 75000, "debt_to_income": 0.35, "credit_score": 720, "loan_amount": 25000},
         {"annual_income": 50000, "debt_to_income": 0.45, "credit_score": 650, "loan_amount": 15000}
       ]
     }'
```

---

## 🛠️ Manual Setup (Advanced Users)

If you prefer to set things up manually:

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv credit-risk-env

# Activate it
# On Mac/Linux:
source credit-risk-env/bin/activate
# On Windows:
credit-risk-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use any text editor
```

### 3. Database Setup
```bash
# Initialize database
python scripts/init_db.py

# Optional: Generate demo data
python scripts/generate_demo_data.py
```

### 4. Start Services Manually
```bash
# Terminal 1 - API Server:
python scripts/run_api.py

# Terminal 2 - Dashboard:
cd dashboard
streamlit run app.py

# Terminal 3 - MLflow (optional):
mlflow ui --host 0.0.0.0 --port 5000
```

---

## ⚙️ Configuration Options

### Environment Variables (.env file):
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./data/riskflow.db

# Real Financial Data APIs
FRED_API_KEY=your_fred_key_here          # Free from fred.stlouisfed.org
TAVILY_API_KEY=your_tavily_key_here      # Paid service ($5-10/month)

# LLM Provider: "openai" or "ollama" or "none"
LLM_PROVIDER=openai

# OpenAI (if using OpenAI)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Ollama (if using Ollama)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Application Settings
ENVIRONMENT=development
DEBUG=true
```

**⚠️ Data Source Behavior:**
- **No API keys**: Uses fallback demo data for basic functionality
- **FRED_API_KEY only**: Real economic data (fed rates, unemployment)
- **TAVILY_API_KEY**: Real-time credit spreads (requires paid subscription)

---

## 🚨 Troubleshooting

### Common Issues:

#### 1. "Port already in use"
```bash
# Find what's using the port
lsof -i :8000  # or :8501

# Kill the process
kill -9 <PID>
```

#### 2. "Python not found"
- Install Python from [python.org](https://python.org)
- Make sure it's in your PATH
- Try `python3` instead of `python`

#### 3. "Permission denied on start-app.sh"
```bash
chmod +x start-app.sh
```

#### 4. "Module not found errors"
```bash
# Make sure virtual environment is activated
source credit-risk-env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 5. "Database errors"
```bash
# Reset database
rm data/riskflow.db
python scripts/init_db.py
```

#### 6. "Empty dashboard boxes"
- Make sure the API server is running (check http://localhost:8000/health)
- Check the terminal for error messages
- Try refreshing the dashboard page

#### 7. "No real market data" errors
- **Solution 1**: Get free FRED API key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
- **Solution 2**: App will work with demo data if no API keys provided
- **Note**: Tavily API is paid ($5-10/month) for credit spread data only

---

## 📚 Understanding the Technology

### What's Under the Hood:

#### **FastAPI (Backend)**
- Serves ML predictions via REST API
- Automatic API documentation at `/docs`
- High-performance async Python framework

#### **Streamlit (Frontend)**  
- Interactive web dashboard
- Real-time charts and metrics
- Point-and-click ML model management

#### **MLflow (ML Management)**
- Tracks model experiments
- Manages model versions
- Stores model artifacts

#### **SQLite (Database)**
- Stores predictions and metrics
- Lightweight, no setup required
- Perfect for development and small deployments

#### **LLM Integration**
- Provides human-readable risk explanations
- Auto-generates model documentation
- Powered by OpenAI GPT or local Ollama models

---

## 🎯 What You Can Build With This

### For Learning:
- Understand MLOps pipelines
- Learn credit risk modeling
- Practice with real ML deployment
- Study production-grade code structure

### For Professionals:
- Portfolio project for job interviews
- Base for hedge fund/bank ML systems
- Template for other ML applications
- Production-ready credit risk platform

### For Businesses:
- Credit scoring for lending
- Risk assessment for investments
- Model monitoring for compliance
- AI-powered risk insights

---

## 🔄 Next Steps After Setup

### 1. Explore the Dashboard
- Click through all tabs
- Try the prediction calculator
- Check system metrics

### 2. Test the API
- Visit http://localhost:8000/docs
- Try the interactive API documentation
- Make test predictions

### 3. Train Your Own Model
```bash
python scripts/train_models.py
```

### 4. Add Your Data
- Place CSV files in `data/raw/`
- Update data processing scripts
- Retrain models with your data

### 5. Customize Features
- Modify prediction logic in `src/models/`
- Add new dashboard pages
- Integrate with your systems

---

## 🤝 Getting Help

### Documentation:
- **API Docs**: http://localhost:8000/docs (when running)
- **Code Comments**: Every file is well-documented
- **Error Messages**: Check terminal output for detailed errors

### Common Resources:
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Community:
- GitHub Issues for bug reports
- Discussions for questions
- Pull Requests for contributions

---

## 🏆 Success Metrics

After setup, you should see:
- ✅ Green "REAL DATA CONNECTED" in dashboard
- ✅ API responding at http://localhost:8000/health
- ✅ Dashboard loading at http://localhost:8501
- ✅ Prediction calculator working
- ✅ System metrics updating in real-time

---

## 📞 Support

If you get stuck:
1. Check the troubleshooting section above
2. Look at terminal output for error messages
3. Try the manual setup process
4. Open a GitHub issue with error details

**Remember**: This is designed to work out of the box. If it doesn't, it's likely a simple configuration issue that can be quickly resolved.

---

**🎉 Congratulations!** You now have a production-grade MLOps platform running locally. This same system can be deployed to cloud providers for real business use.

---

**Built with ❤️ for the future of credit risk modeling**