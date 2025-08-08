#!/bin/bash
# Setup script for ManaMind development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from project root
if [ ! -f "pyproject.toml" ] || [ ! -d "src/manamind" ]; then
    log_error "Please run this script from the ManaMind project root directory"
    exit 1
fi

log_info "Setting up ManaMind development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    log_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

log_success "Python version check passed: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install ManaMind in development mode
log_info "Installing ManaMind in development mode..."
pip install -e .[dev,training]

# Create data directories
log_info "Creating data directories..."
mkdir -p data/{checkpoints,logs,game_logs,cards}
mkdir -p logs
log_success "Data directories created"

# Download Forge if not present
if [ ! -d "forge" ]; then
    log_info "Downloading Forge game engine..."
    
    # Create forge directory
    mkdir -p forge
    
    # Download latest Forge release
    FORGE_URL="https://releases.cardforge.org/forge/forge-gui-latest.tar.bz2"
    
    if command -v wget > /dev/null; then
        wget -O forge/forge.tar.bz2 "$FORGE_URL"
    elif command -v curl > /dev/null; then
        curl -L -o forge/forge.tar.bz2 "$FORGE_URL"
    else
        log_error "Neither wget nor curl found. Please install one of them or download Forge manually."
        exit 1
    fi
    
    # Extract Forge
    log_info "Extracting Forge..."
    tar -xjf forge/forge.tar.bz2 -C forge --strip-components=1
    rm forge/forge.tar.bz2
    
    log_success "Forge downloaded and extracted"
else
    log_info "Forge already exists"
fi

# Check Java installation
if command -v java > /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [ "$JAVA_VERSION" -ge 8 ]; then
        log_success "Java $JAVA_VERSION found"
    else
        log_warning "Java 8 or higher is recommended for Forge. Found: Java $JAVA_VERSION"
    fi
else
    log_warning "Java not found. Please install Java 8 or higher for Forge integration."
fi

# Setup pre-commit hooks if available
if command -v pre-commit > /dev/null; then
    log_info "Setting up pre-commit hooks..."
    pre-commit install
    log_success "Pre-commit hooks installed"
else
    log_info "Pre-commit not available (this is optional)"
fi

# Test ManaMind installation
log_info "Testing ManaMind installation..."
if python -c "import manamind; print('ManaMind version:', manamind.__version__)" 2>/dev/null; then
    log_success "ManaMind installation test passed"
else
    log_error "ManaMind installation test failed"
    exit 1
fi

# Test CLI
log_info "Testing ManaMind CLI..."
if manamind info > /dev/null 2>&1; then
    log_success "ManaMind CLI test passed"
else
    log_error "ManaMind CLI test failed"
    exit 1
fi

# Create example configuration if it doesn't exist
if [ ! -f "configs/local.yaml" ]; then
    log_info "Creating local configuration file..."
    cp configs/base.yaml configs/local.yaml
    log_info "Edit configs/local.yaml to customize your settings"
fi

# Setup complete
log_success "ManaMind development environment setup complete!"
echo
log_info "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Test Forge integration: manamind forge-test"
echo "  3. Start training: manamind train --config configs/local.yaml"
echo "  4. Or start development server: docker-compose --profile development up"
echo
log_info "For more information, see the documentation in docs/"