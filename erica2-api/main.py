"""
ERICA API - Main Application Entry Point
Embryo Ranking and Image Classification Analysis
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from typing import Optional

# Import configuration
from config import config, get_parse_headers, get_s3_config, get_api_key, get_validation_key

# Import API functions
from erica_api import erica_api

# Import startup checks (optional)
try:
    from auto_requirements import startup_requirements_check
    HAS_AUTO_REQUIREMENTS = True
except ImportError:
    HAS_AUTO_REQUIREMENTS = False


# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/erica_{config.environment}.log', mode='a')
    ] if Path('logs').exists() else [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('erica-api')


# ============================================
# SECURITY
# ============================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Verify API key for protected endpoints"""
    expected_key = get_api_key()
    
    # In development, allow access without key
    if config.debug and not api_key:
        logger.warning("API Key not provided, but allowed in debug mode")
        return None
    
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return api_key


# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("=" * 60)
    logger.info(f"  ERICA API Starting - {config.environment.upper()}")
    logger.info("=" * 60)
    logger.info(f"  Environment: {config.environment}")
    logger.info(f"  Version: {config.version}")
    logger.info(f"  API URL: {config.api_url}")
    logger.info(f"  Port: {config.port}")
    logger.info(f"  Debug: {config.debug}")
    logger.info(f"  Parse URL: {config.parse_server_url}")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info("=" * 60)
    
    # Create logs directory if needed
    logs_dir = Path(__file__).parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Run requirements check
    if HAS_AUTO_REQUIREMENTS:
        try:
            startup_requirements_check()
        except Exception as e:
            logger.warning(f"Requirements check failed: {e}")
    
    # Pre-load models in production for faster first request
    if config.environment == 'production':
        try:
            logger.info("Pre-loading models...")
            from utils.erica_cropper import cropper
            from utils.erica_model import erica
            logger.info("Models pre-loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to pre-load models: {e}")
    
    yield  # Application runs
    
    # Shutdown
    logger.info("ERICA API shutting down...")


# ============================================
# FASTAPI APPLICATION
# ============================================
app = FastAPI(
    title="ERICA API",
    description="Embryo Ranking and Image Classification Analysis API",
    version=config.version,
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# MIDDLEWARE
# ============================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Get request info
    method = request.method
    url = str(request.url.path)
    client = request.client.host if request.client else "unknown"
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (time.time() - start_time) * 1000
    
    # Log
    status = response.status_code
    log_msg = f"{method} {url} - {status} - {duration:.2f}ms - {client}"
    
    if status >= 500:
        logger.error(log_msg)
    elif status >= 400:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)
    
    return response


# ============================================
# HEALTH & STATUS ENDPOINTS
# ============================================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ERICA API",
        "version": config.version,
        "environment": config.environment,
        "api_url": config.api_url,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": config.version,
        "environment": config.environment,
        "api_url": config.api_url
    }


@app.get("/status")
async def status(api_key: str = Depends(verify_api_key)):
    """Detailed status endpoint (requires API key)"""
    models_dir = Path(__file__).parent / 'models'
    
    models_status = {
        'scoring': (models_dir / 'erica_model2.pth').exists(),
        'cropper': (models_dir / 'erica_cropper.pt').exists(),
        'segmentor': (models_dir / 'erica_segmentor_n.pt').exists(),
        'scaler': (models_dir / 'scaler_info.json').exists(),
    }
    
    return {
        "status": "running",
        "version": config.version,
        "environment": config.environment,
        "api_url": config.api_url,
        "debug": config.debug,
        "models": models_status,
        "models_ready": all(models_status.values()),
        "parse_configured": bool(os.getenv('PARSE_APPLICATION_ID')),
        "parse_url": config.parse_server_url,
        "s3_configured": bool(os.getenv('ERICA_S3_BUCKET')),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================
# MAIN RANKING ENDPOINT
# ============================================
@app.post("/rankthisone")
async def erica_ranking(item: dict, request: Request, api_key: str = Depends(verify_api_key)):
    """Main embryo ranking endpoint (requires API key)"""
    request_id = f"req_{int(time.time() * 1000)}"
    
    start_time = time.time()
    
    try:
        # Invoke the ERICA pipeline
        result = erica_api(item)
        
        duration = (time.time() - start_time) * 1000
        
        # Check if result indicates an error
        if isinstance(result, dict) and result.get('status', 200) != 200:
            return JSONResponse(
                status_code=result.get('status', 500),
                content={
                    **result,
                    "request_id": request_id,
                    "duration_ms": round(duration, 2)
                }
            )
        
        return {
            "status": 200,
            "message": "Ranking done.",
            "request_id": request_id,
            "duration_ms": round(duration, 2)
        }
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "error": str(e),
                "request_id": request_id,
                "duration_ms": round(duration, 2)
            }
        )


# ============================================
# DEVELOPMENT ENDPOINTS (debug only)
# ============================================
if config.debug:
    @app.get("/debug/config")
    async def debug_config(api_key: str = Depends(verify_api_key)):
        """Show current configuration (debug only)"""
        return {
            "environment": config.environment,
            "api_url": config.api_url,
            "port": config.port,
            "host": config.host,
            "debug": config.debug,
            "reload": config.reload,
            "parse_url": config.parse_server_url,
            "models": config.models,
            "has_api_key": bool(get_api_key()),
            "has_validation_key": bool(get_validation_key())
        }
    
    @app.post("/debug/test-pipeline")
    async def debug_test_pipeline(item: dict, api_key: str = Depends(verify_api_key)):
        """Test pipeline without saving results (debug only)"""
        from utils.get_embryos_db import get_embryos_from_database
        
        object_id = item.get('objectId')
        if not object_id:
            raise HTTPException(status_code=400, detail="objectId required")
        
        embryos, mother_age, oocyte_origin, clinic_id = get_embryos_from_database(object_id)
        return {
            "object_id": object_id,
            "embryos_found": len(embryos) if embryos else 0,
            "mother_age": mother_age,
            "oocyte_origin": oocyte_origin,
            "clinic_id": clinic_id,
            "embryos_sample": embryos[:3] if embryos else []
        }


# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting ERICA API on {config.host}:{config.port}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Parse URL: {config.parse_server_url}")
    
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level="debug" if config.debug else "info"
    )
