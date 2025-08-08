#!/bin/bash

# Local CI Check Script for ManaMind Project
# This script runs the same checks as the CI pipeline locally for faster iteration

set -e  # Exit on any error

echo "🔍 ManaMind Local CI Checks Starting..."
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2 PASSED${NC}"
    else
        echo -e "${RED}❌ $2 FAILED${NC}"
        return 1
    fi
}

# Track overall status
OVERALL_STATUS=0

# 1. MyPy Type Checking (Most Important - Blocks CI)
echo -e "\n${YELLOW}1. Running MyPy Type Checking...${NC}"
if mypy src; then
    print_status 0 "MyPy Type Checking"
else
    print_status 1 "MyPy Type Checking"
    OVERALL_STATUS=1
fi

# 2. Black Code Formatting
echo -e "\n${YELLOW}2. Checking Black Formatting...${NC}"
if black --check --line-length 79 src tests; then
    print_status 0 "Black Formatting"
else
    print_status 1 "Black Formatting"
    OVERALL_STATUS=1
    echo -e "${YELLOW}💡 Fix with: black --line-length 79 src tests${NC}"
fi

# 3. Import Sorting (isort)
echo -e "\n${YELLOW}3. Checking Import Sorting...${NC}"
if isort --check-only src tests; then
    print_status 0 "Import Sorting (isort)"
else
    print_status 1 "Import Sorting (isort)"
    OVERALL_STATUS=1
    echo -e "${YELLOW}💡 Fix with: isort src tests${NC}"
fi

# 4. Linting (flake8)
echo -e "\n${YELLOW}4. Running Linting (flake8)...${NC}"
if flake8 src tests; then
    print_status 0 "Linting (flake8)"
else
    print_status 1 "Linting (flake8)"
    OVERALL_STATUS=1
fi

# 5. Test Suite (if we get this far)
echo -e "\n${YELLOW}5. Running Test Suite...${NC}"
if pytest --cov=src/manamind --cov-report=xml --cov-report=term-missing -v; then
    print_status 0 "Test Suite"
else
    print_status 1 "Test Suite"
    OVERALL_STATUS=1
fi

echo -e "\n========================================"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED! Safe to push to CI.${NC}"
else
    echo -e "${RED}💥 SOME CHECKS FAILED! Fix issues before pushing.${NC}"
    echo -e "\n${YELLOW}Quick Fixes:${NC}"
    echo "  • Format code: black --line-length 79 src tests"
    echo "  • Sort imports: isort src tests"
    echo "  • Fix types: Focus on mypy errors above"
    echo "  • Run this script again after fixes"
fi

echo -e "\n${YELLOW}💡 Pro Tips:${NC}"
echo "  • Run 'mypy src' first - it's the fastest check"
echo "  • Use 'act' to run full CI locally (takes 2-3 minutes)"
echo "  • Set up pre-commit hooks: 'pre-commit install'"

exit $OVERALL_STATUS