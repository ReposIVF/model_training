"""
ERICA API - Centralized Configuration
Manages environment-specific settings for development, staging, and production.

Domain: https://erica.ivf20.app
"""

import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv


class Config:
    """Centralized configuration management for ERICA API"""
    
    def __init__(self):
        self._load_environment()
    
    def _log_env_status(self):
        """Log status of environment variables"""
        required_vars = {
            'PARSE_APPLICATION_ID': os.getenv('PARSE_APPLICATION_ID'),
            'PARSE_REST_API_KEY': os.getenv('PARSE_REST_API_KEY'),
            'ERICA_S3_BUCKET': os.getenv('ERICA_S3_BUCKET'),
            'ERICA_S3_ACCESS_KEY': os.getenv('ERICA_S3_ACCESS_KEY'),
            'ERICA_S3_SECRET_KEY': os.getenv('ERICA_S3_SECRET_KEY'),
            'API_SECRET_KEY': os.getenv('API_SECRET_KEY'),
            'VALIDATION_PASS_KEY': os.getenv('VALIDATION_PASS_KEY'),
        }
        
        optional_vars = {
            'PARSE_SERVER_URL': os.getenv('PARSE_SERVER_URL'),
            'ERICA_S3_REGION': os.getenv('ERICA_S3_REGION'),
            'HOST': os.getenv('HOST'),
            'PORT': os.getenv('PORT'),
        }
        
        print("\n[CONFIG] ============ ENVIRONMENT VARIABLES STATUS ============")
        print(f"[CONFIG] Environment: {os.getenv('ERICA_ENV', 'development')}")
        
        # Check required variables
        found = []
        missing = []
        for key, value in required_vars.items():
            if value:
                found.append(key)
                # Show partial value for security keys
                if 'KEY' in key or 'SECRET' in key:
                    masked = value[:4] + '...' + value[-4:] if len(value) > 8 else '***'
                    print(f"[CONFIG] ✓ {key}: {masked}")
                else:
                    print(f"[CONFIG] ✓ {key}: {value}")
            else:
                missing.append(key)
                print(f"[CONFIG] ✗ {key}: NOT SET")
        
        # Check optional variables
        print("\n[CONFIG] Optional variables:")
        for key, value in optional_vars.items():
            status = f"{value}" if value else "using default"
            print(f"[CONFIG]   {key}: {status}")
        
        # Summary
        print(f"\n[CONFIG] Summary: {len(found)}/{len(required_vars)} required variables set")
        if missing:
            print(f"[CONFIG] ⚠️  Missing required variables: {', '.join(missing)}")
        else:
            print("[CONFIG] ✓ All required variables configured")
        print("[CONFIG] =========================================================\n")
    
    def _load_environment(self):
        """Load environment-specific .env file"""
        env = os.getenv('ERICA_ENV', 'development')
        base_path = Path(__file__).parent
        
        # Try environment-specific file first, then fall back to .env
        env_file = base_path / f'.env.{env}'
        if env_file.exists():
            load_dotenv(env_file)
            print(f"[CONFIG] Loaded {env_file.name}")
        else:
            default_env = base_path / '.env'
            if default_env.exists():
                load_dotenv(default_env)
                print(f"[CONFIG] Loaded .env (default)")
            else:
                print(f"[CONFIG] No .env file found, using environment variables")
        
        # Log environment variables status
        self._log_env_status()
    
    @property
    def environment(self) -> str:
        return os.getenv('ERICA_ENV', 'development')
    
    @property
    def version(self) -> str:
        return os.getenv('ERICA_VERSION', '2.0.0')
    
    @property
    def debug(self) -> bool:
        return self.environment in ['development', 'dev']
    
    @property
    def api_url(self) -> str:
        """Public API URL"""
        defaults = {
            'production': 'https://erica.ivf20.app',
            'staging': 'https://erica.ivf20.app/staging',
            'development': 'http://localhost:8001'
        }
        return os.getenv('API_URL', defaults.get(self.environment, defaults['development']))
    
    @property
    def host(self) -> str:
        return os.getenv('HOST', '0.0.0.0')
    
    @property
    def port(self) -> int:
        defaults = {
            'production': 8000,
            'staging': 8002,
            'development': 8001
        }
        return int(os.getenv('PORT', defaults.get(self.environment, 8001)))
    
    @property
    def reload(self) -> bool:
        return self.environment in ['development', 'dev']
    
    # Parse Server Configuration
    @property
    def parse_server_url(self) -> str:
        return os.getenv('PARSE_SERVER_URL', 'https://dish-s.ivf20.app/db')
    
    @property
    def parse_application_id(self) -> str:
        return os.getenv('PARSE_APPLICATION_ID', '')
    
    @property
    def parse_rest_api_key(self) -> str:
        return os.getenv('PARSE_REST_API_KEY', '')
    
    # S3 Configuration
    @property
    def s3_bucket(self) -> str:
        return os.getenv('ERICA_S3_BUCKET', '')
    
    @property
    def s3_region(self) -> str:
        return os.getenv('ERICA_S3_REGION', 'us-east-1')
    
    @property
    def s3_access_key(self) -> str:
        return os.getenv('ERICA_S3_ACCESS_KEY', '')
    
    @property
    def s3_secret_key(self) -> str:
        return os.getenv('ERICA_S3_SECRET_KEY', '')
    
    # Model paths
    @property
    def models(self) -> Dict[str, str]:
        base = Path(__file__).parent / 'models'
        return {
            'scoring': str(base / os.getenv('MODEL_SCORING', 'erica_model2.pth')),
            'cropper': str(base / os.getenv('MODEL_CROPPER', 'erica_cropper.pt')),
            'segmentor': str(base / os.getenv('MODEL_SEGMENTOR', 'erica_segmentor_n.pt')),
            'scaler': str(base / os.getenv('MODEL_SCALER', 'scaler_info.json')),
        }
    
    def summary(self) -> Dict:
        """Get configuration summary (safe for logging)"""
        return {
            'environment': self.environment,
            'version': self.version,
            'api_url': self.api_url,
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            'parse_url': self.parse_server_url,
            'parse_configured': bool(self.parse_application_id),
            's3_configured': bool(self.s3_bucket),
        }


# Global config instance
config = Config()


def get_parse_headers() -> Dict[str, str]:
    """Get headers for Parse Server requests"""
    return {
        'X-Parse-Application-Id': config.parse_application_id,
        'X-Parse-REST-API-Key': config.parse_rest_api_key,
        'Content-Type': 'application/json'
    }


def get_s3_config() -> Dict[str, str]:
    """Get S3 configuration for boto3"""
    return {
        'bucket': config.s3_bucket,
        'region': config.s3_region,
        'aws_access_key_id': config.s3_access_key,
        'aws_secret_access_key': config.s3_secret_key,
    }


def get_api_key() -> str:
    """Get API authentication key"""
    return os.getenv('API_SECRET_KEY', '')


def get_validation_key() -> str:
    """Get validation key for ranking operations"""
    return os.getenv('VALIDATION_PASS_KEY', '')


# CLI usage
if __name__ == '__main__':
    import json
    print("\n" + "=" * 50)
    print("ERICA API Configuration")
    print("=" * 50)
    print(json.dumps(config.summary(), indent=2))
    print("=" * 50)
