#!/bin/bash

# RiskFlow MLOps Platform - Easy Startup Script
# This script will start both the FastAPI backend and Streamlit frontend

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 1
    else
        return 0
    fi
}

# Function to wait for service
wait_for_service() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service to start on port $port..."
    
    while ! curl -s http://localhost:$port/health >/dev/null 2>&1; do
        if [ $attempt -eq $max_attempts ]; then
            print_error "$service failed to start after $max_attempts attempts"
            return 1
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_success "$service is running on port $port"
    return 0
}

# Function to cleanup background processes
cleanup() {
    print_status "Shutting down services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
        print_status "FastAPI server stopped"
    fi
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null || true
        print_status "Streamlit dashboard stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_status "🚀 Starting RiskFlow MLOps Platform..."
echo

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists python; then
    print_error "Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command_exists pip; then
    print_error "pip is not installed. Please install pip first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required. You have $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Check if virtual environment should be created
if [ ! -d "credit-risk-env" ]; then
    print_status "Creating virtual environment..."
    python -m venv credit-risk-env
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source credit-risk-env/Scripts/activate
else
    source credit-risk-env/bin/activate
fi
print_success "Virtual environment activated"

# Install dependencies
print_status "Installing/updating dependencies..."
pip install -r requirements.txt
print_success "Dependencies installed"

# Check for .env file
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating template..."
    cat > .env << 'EOF'
# Real Financial Data APIs
FRED_API_KEY=your_fred_api_key_here          # Free from fred.stlouisfed.org
TAVILY_API_KEY=your_tavily_api_key_here      # Paid service for credit spreads

# OpenAI Configuration (Required for LLM features)
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///./data/riskflow.db

# Application Settings
ENVIRONMENT=development
DEBUG=true

# LLM Provider: "openai" or "ollama"
LLM_PROVIDER=openai
EOF
    print_warning "Please edit .env file with your API keys:"
    echo "  - FRED_API_KEY: Free from fred.stlouisfed.org (for real economic data)"
    echo "  - OPENAI_API_KEY: For AI features (or use free Ollama)"
    echo "  - TAVILY_API_KEY: Paid service for credit spreads (optional)"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data logs data/models data/cache data/raw data/processed mlflow
print_success "Directories created"

# Initialize database
print_status "Initializing database..."
python scripts/init_db.py
print_success "Database initialized"

# Check ports
print_status "Checking port availability..."

if ! check_port 8000; then
    print_error "Port 8000 is already in use. Please stop the service using this port."
    exit 1
fi

if ! check_port 8501; then
    print_error "Port 8501 is already in use. Please stop the service using this port."
    exit 1
fi

print_success "Ports 8000 and 8501 are available"

# Start FastAPI server
print_status "Starting FastAPI server..."
python scripts/run_api.py &
API_PID=$!

# Wait for API to be ready
if wait_for_service 8000 "FastAPI server"; then
    print_success "✅ FastAPI server is running at http://localhost:8000"
    print_status "   📖 API Documentation: http://localhost:8000/docs"
else
    print_error "Failed to start FastAPI server"
    cleanup
    exit 1
fi

# Start Streamlit dashboard
print_status "Starting Streamlit dashboard..."
cd dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
DASHBOARD_PID=$!
cd ..

# Wait a bit for Streamlit to start
sleep 5

print_success "✅ Streamlit dashboard is starting at http://localhost:8501"

echo
print_success "🎉 RiskFlow MLOps Platform is now running!"
echo
echo -e "${GREEN}📊 Dashboard:${NC} http://localhost:8501"
echo -e "${GREEN}🔧 API:${NC} http://localhost:8000"
echo -e "${GREEN}📖 API Docs:${NC} http://localhost:8000/docs"
echo
print_status "Press Ctrl+C to stop all services"
echo

# Keep script running and wait for signals
wait