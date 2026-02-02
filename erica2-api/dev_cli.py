#!/usr/bin/env python3
"""
ERICA Development CLI - Main Command Line Interface
Comprehensive tool for development, testing, and deployment.

Usage:
    python dev_cli.py [command] [options]

Commands:
    env         - Environment management
    health      - Health checks
    models      - Model testing
    rank        - Local ranking
    start       - Start server
    deploy      - Deployment tools
    logs        - Log management
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import config, get_api_key, get_validation_key
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False
    print("[WARN] Config not loaded, some features unavailable")


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warn(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


# ============================================
# ENVIRONMENT COMMANDS
# ============================================
def cmd_env_show():
    """Show current environment configuration"""
    print_header("Environment Configuration")
    
    if not CONFIG_LOADED:
        print_error("Config not loaded")
        return
    
    summary = config.summary()
    for key, value in summary.items():
        status = Colors.GREEN if value else Colors.YELLOW
        print(f"  {Colors.BOLD}{key}:{Colors.END} {status}{value}{Colors.END}")
    
    # Check API keys
    print(f"\n  {Colors.BOLD}Security:{Colors.END}")
    api_key = get_api_key()
    val_key = get_validation_key()
    print(f"    API Key: {'✓ configured' if api_key else '✗ missing'}")
    print(f"    Validation Key: {'✓ configured' if val_key else '✗ missing'}")


def cmd_env_set(env_name: str):
    """Set environment"""
    valid = ['development', 'staging', 'production', 'dev', 'prod']
    if env_name not in valid:
        print_error(f"Invalid environment. Use: {', '.join(valid)}")
        return
    
    os.environ['ERICA_ENV'] = env_name
    print_success(f"Environment set to: {env_name}")
    print_info(f"Export: export ERICA_ENV={env_name}")


def cmd_env_validate():
    """Validate environment configuration"""
    print_header("Environment Validation")
    
    checks = [
        ('ERICA_ENV', os.getenv('ERICA_ENV')),
        ('PARSE_SERVER_URL', os.getenv('PARSE_SERVER_URL')),
        ('PARSE_APPLICATION_ID', os.getenv('PARSE_APPLICATION_ID')),
        ('PARSE_REST_API_KEY', os.getenv('PARSE_REST_API_KEY')),
        ('API_SECRET_KEY', os.getenv('API_SECRET_KEY')),
        ('VALIDATION_PASS_KEY', os.getenv('VALIDATION_PASS_KEY')),
        ('ERICA_S3_BUCKET', os.getenv('ERICA_S3_BUCKET')),
    ]
    
    all_valid = True
    for name, value in checks:
        if value:
            print_success(f"{name}: configured")
        else:
            print_warn(f"{name}: not set")
            all_valid = False
    
    # Check conda environment
    print(f"\n{Colors.BOLD}Conda Environment:{Colors.END}")
    conda_env = os.getenv('CONDA_DEFAULT_ENV')
    if conda_env:
        print_success(f"Conda environment: {conda_env}")
        conda_prefix = os.getenv('CONDA_PREFIX')
        if conda_prefix:
            print(f"    Prefix: {conda_prefix}")
    else:
        print_info("Not using conda environment")
    
    # Check models
    print(f"\n{Colors.BOLD}Models:{Colors.END}")
    models_dir = Path(__file__).parent / 'models'
    model_files = [
        'erica_model2.pth',
        'erica_cropper.pt',
        'erica_segmentor_n.pt',
        'scaler_info.json'
    ]
    
    for model in model_files:
        path = models_dir / model
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print_success(f"{model}: {size:.2f} MB")
        else:
            print_error(f"{model}: not found")
            all_valid = False
    
    return all_valid


# ============================================
# HEALTH CHECK COMMANDS
# ============================================
def cmd_health_local():
    """Check local API health"""
    import requests
    
    print_header("Local Health Check")
    
    port = config.port if CONFIG_LOADED else 8001
    url = f"http://localhost:{port}"
    
    try:
        # Root endpoint
        resp = requests.get(f"{url}/", timeout=5)
        if resp.status_code == 200:
            print_success(f"Root endpoint: OK")
            print(f"    Response: {resp.json()}")
        else:
            print_error(f"Root endpoint: {resp.status_code}")
        
        # Health endpoint
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            print_success(f"Health endpoint: OK")
        else:
            print_error(f"Health endpoint: {resp.status_code}")
            
    except requests.ConnectionError:
        print_error(f"Cannot connect to {url}")
        print_info("Is the server running? Try: python dev_cli.py start")
    except Exception as e:
        print_error(f"Error: {e}")


def cmd_health_remote(env: str = 'production'):
    """Check remote API health"""
    import requests
    
    print_header(f"Remote Health Check ({env})")
    
    urls = {
        'production': 'https://erica.ivf20.app',
        'staging': 'https://erica.ivf20.app/staging',
    }
    
    url = urls.get(env, urls['production'])
    
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        if resp.status_code == 200:
            print_success(f"Remote API healthy")
            data = resp.json()
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print_error(f"Health check failed: {resp.status_code}")
    except Exception as e:
        print_error(f"Cannot reach {url}: {e}")


# ============================================
# MODEL COMMANDS
# ============================================
def cmd_models_list():
    """List available models"""
    print_header("Available Models")
    
    models_dir = Path(__file__).parent / 'models'
    
    if not models_dir.exists():
        print_error("Models directory not found")
        return
    
    for file in sorted(models_dir.iterdir()):
        if file.is_file():
            size = file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"  {Colors.BOLD}{file.name}{Colors.END}")
            print(f"    Size: {size:.2f} MB")
            print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")


def cmd_models_test():
    """Test model loading"""
    print_header("Model Loading Test")
    
    import time
    
    # Test cropper
    print_info("Loading cropper model...")
    start = time.time()
    try:
        from utils.erica_cropper import cropper
        print_success(f"Cropper loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print_error(f"Cropper failed: {e}")
    
    # Test scoring model
    print_info("Loading scoring model...")
    start = time.time()
    try:
        from utils.erica_model import erica
        print_success(f"Scoring model loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print_error(f"Scoring model failed: {e}")
    
    # Test segmentor
    print_info("Loading segmentor model...")
    start = time.time()
    try:
        from utils.model_explorer import get_segmentation_model_weights
        print_success(f"Segmentor info loaded in {time.time() - start:.2f}s")
    except Exception as e:
        print_error(f"Segmentor failed: {e}")


def cmd_models_benchmark():
    """Benchmark model inference"""
    print_header("Model Benchmark")
    
    import time
    import numpy as np
    
    try:
        from PIL import Image
        import torch
    except ImportError:
        print_error("Required packages not installed (PIL, torch)")
        return
    
    # Create dummy image
    print_info("Creating test image...")
    img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    
    # Benchmark cropper
    print_info("Benchmarking cropper (5 iterations)...")
    try:
        from ultralytics import YOLO
        model_path = Path(__file__).parent / 'models' / 'erica_cropper.pt'
        if model_path.exists():
            model = YOLO(str(model_path))
            times = []
            for _ in range(5):
                start = time.time()
                _ = model.predict(img, verbose=False)
                times.append(time.time() - start)
            avg = sum(times) / len(times)
            print_success(f"Cropper avg: {avg * 1000:.1f}ms per image")
        else:
            print_warn("Cropper model not found")
    except Exception as e:
        print_error(f"Cropper benchmark failed: {e}")


# ============================================
# RANKING COMMANDS
# ============================================
def cmd_rank_local(image_path: str, mother_age: int = 35):
    """Run local ranking on an image"""
    print_header("Local Image Ranking")
    
    path = Path(image_path)
    if not path.exists():
        print_error(f"Image not found: {image_path}")
        return
    
    print_info(f"Processing: {path.name}")
    
    try:
        from PIL import Image
        from utils.erica_cropper import cropper
        from utils.erica_img_process import process_imagesv2
        from utils.erica_model import erica
        from utils.model_explorer import get_scaler_info
        
        # Mock embryo data
        embryo = {
            'embryo': 'local_test',
            'image': str(path),
            'local_path': str(path),
            'isEmbryo': True,
            'pgt': ''
        }
        
        print_info("Running cropper...")
        embryos = cropper([embryo], {})
        
        if not embryos:
            print_error("No embryo detected in image")
            return
        
        print_success(f"Embryo detected")
        
        print_info("Extracting features...")
        scaler_path = get_scaler_info({})
        df = process_imagesv2(embryos, mother_age=mother_age, scaler_json_path=scaler_path)
        
        print_info("Running scoring model...")
        ranked = erica(df, {})
        
        print_success("Ranking complete!")
        for emb in ranked:
            print(f"\n  Score: {Colors.BOLD}{emb.get('score', 'N/A')}{Colors.END}")
            print(f"  Embryo: {emb.get('embryo', 'N/A')}")
            
    except Exception as e:
        print_error(f"Ranking failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_rank_batch(folder_path: str):
    """Run batch ranking on a folder of images"""
    print_header("Batch Image Ranking")
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print_error(f"Folder not found: {folder_path}")
        return
    
    images = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg'))
    
    if not images:
        print_error("No images found in folder")
        return
    
    print_info(f"Found {len(images)} images")
    
    for img in images:
        print(f"\n{Colors.BOLD}{img.name}{Colors.END}")
        cmd_rank_local(str(img))


def cmd_rank_remote(object_id: str, env: str = 'development'):
    """Trigger remote ranking"""
    import requests
    
    print_header(f"Remote Ranking ({env})")
    
    urls = {
        'production': 'https://erica.ivf20.app',
        'staging': 'https://erica.ivf20.app/staging',
        'development': f'http://localhost:{config.port if CONFIG_LOADED else 8001}'
    }
    
    url = urls.get(env, urls['development'])
    api_key = get_api_key() if CONFIG_LOADED else os.getenv('API_SECRET_KEY', '')
    val_key = get_validation_key() if CONFIG_LOADED else os.getenv('VALIDATION_PASS_KEY', '')
    
    print_info(f"Target: {url}")
    print_info(f"Object ID: {object_id}")
    
    try:
        resp = requests.post(
            f"{url}/rankthisone",
            headers={
                'Content-Type': 'application/json',
                'X-API-Key': api_key
            },
            json={
                'objectId': object_id,
                'validation_key': val_key
            },
            timeout=300
        )
        
        if resp.status_code == 200:
            print_success("Ranking completed!")
            print(json.dumps(resp.json(), indent=2))
        else:
            print_error(f"Ranking failed: {resp.status_code}")
            print(resp.text)
            
    except Exception as e:
        print_error(f"Request failed: {e}")


# ============================================
# SERVER COMMANDS
# ============================================
def cmd_start(env: str = 'development', port: int = None):
    """Start the API server"""
    print_header(f"Starting ERICA API ({env})")
    
    os.environ['ERICA_ENV'] = env
    
    if port:
        actual_port = port
    elif CONFIG_LOADED:
        actual_port = config.port
    else:
        actual_port = {'development': 8001, 'staging': 8002, 'production': 8000}.get(env, 8001)
    
    print_info(f"Environment: {env}")
    print_info(f"Port: {actual_port}")
    print_info(f"URL: http://localhost:{actual_port}")
    print_info("Press Ctrl+C to stop\n")
    
    reload_flag = '--reload' if env == 'development' else ''
    
    cmd = f"uvicorn main:app --host 0.0.0.0 --port {actual_port} {reload_flag}"
    os.system(cmd)


def cmd_start_pm2(env: str = 'production'):
    """Start with PM2"""
    print_header(f"Starting with PM2 ({env})")
    
    app_name = f"erica-{env}"
    port = {'development': 8001, 'staging': 8002, 'production': 8000}.get(env, 8000)
    
    cmd = f"""pm2 start "uvicorn main:app --host 0.0.0.0 --port {port}" --name {app_name} --interpreter python3"""
    
    print_info(f"Command: {cmd}")
    os.system(cmd)


# ============================================
# LOGS COMMANDS
# ============================================
def cmd_requirements():
    """Update requirements"""
    print_info("Updating requirements.txt...")
    os.system("python3 auto_requirements.py")


def cmd_conda_export():
    """Export conda environment"""
    print_info("Exporting conda environment...")
    os.system("python3 auto_requirements.py --conda")


# ============================================
# LOGS COMMANDS
# ============================================
    
    logs_dir = Path(__file__).parent / 'logs'
    
    if not logs_dir.exists():
        print_warn("Logs directory not found")
        return
    
    log_files = list(logs_dir.glob('*.log'))
    
    if not log_files:
        print_warn("No log files found")
        return
    
    # Get most recent
    latest = max(log_files, key=lambda p: p.stat().st_mtime)
    print_info(f"Showing: {latest.name}")
    
    with open(latest) as f:
        content = f.readlines()
        for line in content[-lines:]:
            print(line.rstrip())


def cmd_logs_tail(env: str = None):
    """Tail logs in real-time"""
    logs_dir = Path(__file__).parent / 'logs'
    
    if env:
        log_file = logs_dir / f'erica_{env}.log'
    else:
        log_files = list(logs_dir.glob('*.log'))
        if not log_files:
            print_error("No log files found")
            return
        log_file = max(log_files, key=lambda p: p.stat().st_mtime)
    
    print_info(f"Tailing: {log_file}")
    os.system(f"tail -f {log_file}")


def cmd_logs_clear():
    """Clear all logs"""
    logs_dir = Path(__file__).parent / 'logs'
    
    if not logs_dir.exists():
        return
    
    for log_file in logs_dir.glob('*.log'):
        log_file.unlink()
        print_success(f"Deleted: {log_file.name}")


# ============================================
# MAIN CLI
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description='ERICA Development CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dev_cli.py env show           Show environment config
  python dev_cli.py env set staging    Set environment to staging
  python dev_cli.py health local       Check local API health
  python dev_cli.py health remote      Check production API health
  python dev_cli.py models list        List available models
  python dev_cli.py models test        Test model loading
  python dev_cli.py rank local img.jpg Run local ranking
  python dev_cli.py rank remote ABC123 Trigger remote ranking
  python dev_cli.py start              Start development server
  python dev_cli.py start --env prod   Start production server
  python dev_cli.py logs show          Show recent logs
  python dev_cli.py logs tail          Tail logs in real-time
  python dev_cli.py requirements       Update requirements.txt
  python dev_cli.py conda              Export conda environment
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # ENV commands
    env_parser = subparsers.add_parser('env', help='Environment management')
    env_sub = env_parser.add_subparsers(dest='env_cmd')
    env_sub.add_parser('show', help='Show configuration')
    env_set = env_sub.add_parser('set', help='Set environment')
    env_set.add_argument('name', choices=['development', 'staging', 'production', 'dev', 'prod'])
    env_sub.add_parser('validate', help='Validate configuration')
    
    # HEALTH commands
    health_parser = subparsers.add_parser('health', help='Health checks')
    health_sub = health_parser.add_subparsers(dest='health_cmd')
    health_sub.add_parser('local', help='Check local API')
    health_remote = health_sub.add_parser('remote', help='Check remote API')
    health_remote.add_argument('--env', default='production', choices=['production', 'staging'])
    
    # MODELS commands
    models_parser = subparsers.add_parser('models', help='Model management')
    models_sub = models_parser.add_subparsers(dest='models_cmd')
    models_sub.add_parser('list', help='List models')
    models_sub.add_parser('test', help='Test model loading')
    models_sub.add_parser('benchmark', help='Benchmark models')
    
    # RANK commands
    rank_parser = subparsers.add_parser('rank', help='Ranking operations')
    rank_sub = rank_parser.add_subparsers(dest='rank_cmd')
    rank_local = rank_sub.add_parser('local', help='Local ranking')
    rank_local.add_argument('image', help='Image path')
    rank_local.add_argument('--age', type=int, default=35, help='Mother age (default: 35)')
    rank_batch = rank_sub.add_parser('batch', help='Batch ranking')
    rank_batch.add_argument('folder', help='Folder path')
    rank_remote = rank_sub.add_parser('remote', help='Remote ranking')
    rank_remote.add_argument('object_id', help='Cycle object ID')
    rank_remote.add_argument('--env', default='development', choices=['development', 'staging', 'production'])
    
    # START commands
    start_parser = subparsers.add_parser('start', help='Start server')
    start_parser.add_argument('--env', '-e', default='development', choices=['development', 'staging', 'production'])
    start_parser.add_argument('--port', '-p', type=int, help='Override port')
    start_parser.add_argument('--pm2', action='store_true', help='Start with PM2')
    
    # LOGS commands
    logs_parser = subparsers.add_parser('logs', help='Log management')
    logs_sub = logs_parser.add_subparsers(dest='logs_cmd')
    logs_show = logs_sub.add_parser('show', help='Show logs')
    logs_show.add_argument('-n', '--lines', type=int, default=50, help='Number of lines')
    logs_tail = logs_sub.add_parser('tail', help='Tail logs')
    logs_tail.add_argument('--env', help='Environment to tail')
    logs_sub.add_parser('clear', help='Clear logs')
    
    # REQUIREMENTS commands
    req_parser = subparsers.add_parser('requirements', help='Update requirements')
    
    # CONDA commands
    conda_parser = subparsers.add_parser('conda', help='Export conda environment')
    
    args = parser.parse_args()
    
    # Route commands
    if args.command == 'env':
        if args.env_cmd == 'show':
            cmd_env_show()
        elif args.env_cmd == 'set':
            cmd_env_set(args.name)
        elif args.env_cmd == 'validate':
            cmd_env_validate()
        else:
            cmd_env_show()
    
    elif args.command == 'health':
        if args.health_cmd == 'local':
            cmd_health_local()
        elif args.health_cmd == 'remote':
            cmd_health_remote(args.env)
        else:
            cmd_health_local()
    
    elif args.command == 'models':
        if args.models_cmd == 'list':
            cmd_models_list()
        elif args.models_cmd == 'test':
            cmd_models_test()
        elif args.models_cmd == 'benchmark':
            cmd_models_benchmark()
        else:
            cmd_models_list()
    
    elif args.command == 'rank':
        if args.rank_cmd == 'local':
            cmd_rank_local(args.image, mother_age=args.age)
        elif args.rank_cmd == 'batch':
            cmd_rank_batch(args.folder)
        elif args.rank_cmd == 'remote':
            cmd_rank_remote(args.object_id, args.env)
        else:
            parser.print_help()
    
    elif args.command == 'start':
        if args.pm2:
            cmd_start_pm2(args.env)
        else:
            cmd_start(args.env, args.port)
    
    elif args.command == 'logs':
        if args.logs_cmd == 'show':
            cmd_logs_show(args.lines)
        elif args.logs_cmd == 'tail':
            cmd_logs_tail(args.env)
        elif args.logs_cmd == 'clear':
            cmd_logs_clear()
        else:
            cmd_logs_show()
    
    elif args.command == 'requirements':
        cmd_requirements()
    
    elif args.command == 'conda':
        cmd_conda_export()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
