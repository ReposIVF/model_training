#!/bin/bash
# ============================================
# ERICA API - Quick Scripts
# ============================================
# Source this file: source scripts.sh
# Or run directly: ./scripts.sh [command]
# ============================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================
# Environment Functions
# ============================================

erica-env() {
    echo -e "${BLUE}Current environment:${NC} ${ERICA_ENV:-not set}"
    echo -e "${BLUE}Available environments:${NC} development, staging, production"
}

erica-dev() {
    export ERICA_ENV=development
    echo -e "${GREEN}✓ Environment set to development${NC}"
}

erica-staging() {
    export ERICA_ENV=staging
    echo -e "${GREEN}✓ Environment set to staging${NC}"
}

erica-prod() {
    export ERICA_ENV=production
    echo -e "${YELLOW}⚠ Environment set to production${NC}"
}

# ============================================
# Server Functions
# ============================================

erica-start() {
    local env=${1:-development}
    local port
    
    case $env in
        development|dev) port=8001 ;;
        staging) port=8002 ;;
        production|prod) port=8000 ;;
        *) port=8001 ;;
    esac
    
    echo -e "${BLUE}Starting ERICA API ($env) on port $port...${NC}"
    cd "$SCRIPT_DIR" && ERICA_ENV=$env uvicorn main:app --host 0.0.0.0 --port $port --reload
}

erica-start-pm2() {
    local env=${1:-production}
    local port
    local name
    
    case $env in
        development|dev) port=8001; name="erica-dev" ;;
        staging) port=8002; name="erica-staging" ;;
        production|prod) port=8000; name="erica-prod" ;;
        *) port=8000; name="erica-prod" ;;
    esac
    
    echo -e "${BLUE}Starting ERICA API ($env) with PM2...${NC}"
    cd "$SCRIPT_DIR" && ERICA_ENV=$env pm2 start "uvicorn main:app --host 0.0.0.0 --port $port" --name $name --interpreter python3
}

erica-stop() {
    local env=${1:-all}
    
    if [ "$env" = "all" ]; then
        pm2 stop erica-dev erica-staging erica-prod 2>/dev/null
        echo -e "${GREEN}✓ Stopped all ERICA instances${NC}"
    else
        local name="erica-$env"
        pm2 stop $name 2>/dev/null
        echo -e "${GREEN}✓ Stopped $name${NC}"
    fi
}

erica-restart() {
    local env=${1:-all}
    
    if [ "$env" = "all" ]; then
        pm2 restart erica-dev erica-staging erica-prod 2>/dev/null
    else
        local name="erica-$env"
        pm2 restart $name 2>/dev/null
    fi
}

erica-status() {
    pm2 status
}

erica-logs() {
    local env=${1:-prod}
    local name="erica-$env"
    pm2 logs $name
}

# ============================================
# Development Functions
# ============================================

erica-test() {
    echo -e "${BLUE}Running model tests...${NC}"
    cd "$SCRIPT_DIR" && python model_tester.py test
}

erica-benchmark() {
    echo -e "${BLUE}Running benchmarks...${NC}"
    cd "$SCRIPT_DIR" && python model_tester.py benchmark
}

erica-requirements() {
    echo -e "${BLUE}Updating requirements.txt...${NC}"
    cd "$SCRIPT_DIR" && python3 auto_requirements.py
}

erica-conda() {
    echo -e "${BLUE}Exporting conda environment...${NC}"
    cd "$SCRIPT_DIR" && python3 auto_requirements.py --conda
}

erica-config() {
    echo -e "${BLUE}Configuration:${NC}"
    cd "$SCRIPT_DIR" && python3 config.py
}

# ============================================
# API Testing Functions
# ============================================

erica-health() {
    local port=${1:-8001}
    echo -e "${BLUE}Health check on port $port...${NC}"
    curl -s "http://localhost:$port/health" | python -m json.tool
}

erica-health-prod() {
    echo -e "${BLUE}Health check on production...${NC}"
    curl -s "https://erica.ivf20.app/health" | python -m json.tool
}

erica-rank() {
    local object_id=$1
    local api_key=$2
    local port=${3:-8001}
    
    if [ -z "$object_id" ]; then
        echo -e "${RED}Usage: erica-rank <objectId> [apiKey] [port]${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Triggering ranking for $object_id...${NC}"
    curl -X POST "http://localhost:$port/rankthisone" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${api_key:-dev_secret_key_12345}" \
        -d "{\"objectId\": \"$object_id\", \"validation_key\": \"dev_validation_key_12345\"}"
}

# ============================================
# Deployment Functions
# ============================================

erica-deploy() {
    local env=${1:-staging}
    echo -e "${BLUE}Deploying to $env...${NC}"
    cd "$SCRIPT_DIR" && python deploy.py deploy $env
}

erica-version() {
    cd "$SCRIPT_DIR" && python version_manager.py show
}

erica-bump() {
    local type=${1:-patch}
    cd "$SCRIPT_DIR" && python version_manager.py bump $type
}

# ============================================
# Help
# ============================================

erica-help() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       ERICA API - Quick Commands Reference             ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Environment:${NC}"
    echo "  erica-env              Show current environment"
    echo "  erica-dev              Set to development"
    echo "  erica-staging          Set to staging"
    echo "  erica-prod             Set to production"
    echo ""
    echo -e "${GREEN}Server:${NC}"
    echo "  erica-start [env]      Start server (dev/staging/prod)"
    echo "  erica-start-pm2 [env]  Start with PM2"
    echo "  erica-stop [env|all]   Stop server(s)"
    echo "  erica-restart [env]    Restart server"
    echo "  erica-status           Show PM2 status"
    echo "  erica-logs [env]       View logs"
    echo ""
    echo -e "${GREEN}Development:${NC}"
    echo "  erica-test             Run model tests"
    echo "  erica-benchmark        Run benchmarks"
    echo "  erica-requirements     Update requirements.txt"
    echo "  erica-conda            Export conda environment"
    echo "  erica-config           Show configuration"
    echo ""
    echo -e "${GREEN}API Testing:${NC}"
    echo "  erica-health [port]    Health check"
    echo "  erica-health-prod      Health check production"
    echo "  erica-rank <id> [key]  Trigger ranking"
    echo ""
    echo -e "${GREEN}Deployment:${NC}"
    echo "  erica-deploy [env]     Deploy to environment"
    echo "  erica-version          Show version"
    echo "  erica-bump [type]      Bump version (patch/minor/major)"
    echo ""
}

# ============================================
# Main (when run directly)
# ============================================

if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    case "${1:-help}" in
        help) erica-help ;;
        env) erica-env ;;
        dev) erica-dev ;;
        staging) erica-staging ;;
        prod) erica-prod ;;
        start) erica-start "$2" ;;
        start-pm2) erica-start-pm2 "$2" ;;
        stop) erica-stop "$2" ;;
        restart) erica-restart "$2" ;;
        status) erica-status ;;
        logs) erica-logs "$2" ;;
        test) erica-test ;;
        benchmark) erica-benchmark ;;
        reqda) erica-conda ;;
        conuirements) erica-requirements ;;
        config) erica-config ;;
        health) erica-health "$2" ;;
        health-prod) erica-health-prod ;;
        rank) erica-rank "$2" "$3" "$4" ;;
        deploy) erica-deploy "$2" ;;
        version) erica-version ;;
        bump) erica-bump "$2" ;;
        *) 
            echo -e "${RED}Unknown command: $1${NC}"
            erica-help
            ;;
    esac
fi
