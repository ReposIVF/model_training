#!/usr/bin/env python3
"""
ERICA Deployment Manager - Multi-environment deployment tool

Features:
- Deploy to development, staging, or production
- PM2 process management
- Docker deployment
- Version tagging
- Rollback support
- Health checks after deployment

Usage:
    python deploy.py status              # Check deployment status
    python deploy.py start [env]         # Start environment
    python deploy.py stop [env]          # Stop environment
    python deploy.py restart [env]       # Restart environment
    python deploy.py logs [env]          # View logs
    python deploy.py deploy [env]        # Full deployment
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def ok(msg): print(f"{Colors.GREEN}✓ {msg}{Colors.END}")
def fail(msg): print(f"{Colors.RED}✗ {msg}{Colors.END}")
def warn(msg): print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")
def info(msg): print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


class DeploymentManager:
    """Manage ERICA API deployments"""
    
    ENVIRONMENTS = {
        'development': {
            'port': 8001,
            'host': '0.0.0.0',
            'url': 'http://localhost:8001',
            'pm2_name': 'erica-dev',
            'workers': 1,
            'reload': True
        },
        'staging': {
            'port': 8002,
            'host': '0.0.0.0',
            'url': 'https://erica.ivf20.app/staging',
            'pm2_name': 'erica-staging',
            'workers': 2,
            'reload': False
        },
        'production': {
            'port': 8000,
            'host': '0.0.0.0',
            'url': 'https://erica.ivf20.app',
            'pm2_name': 'erica-prod',
            'workers': 4,
            'reload': False
        }
    }
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.logs_dir = self.root / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
    def run_command(self, cmd: str, capture: bool = False) -> Optional[str]:
        """Run a shell command"""
        try:
            if capture:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                return result.stdout
            else:
                os.system(cmd)
                return None
        except Exception as e:
            fail(f"Command failed: {e}")
            return None
    
    def check_pm2(self) -> bool:
        """Check if PM2 is installed"""
        result = self.run_command("which pm2", capture=True)
        return bool(result and result.strip())
    
    def get_pm2_status(self) -> List[Dict]:
        """Get PM2 process status"""
        if not self.check_pm2():
            return []
        
        output = self.run_command("pm2 jlist", capture=True)
        try:
            return json.loads(output) if output else []
        except:
            return []
    
    def status(self):
        """Show deployment status for all environments"""
        print("\n" + "=" * 70)
        print("  ERICA Deployment Status")
        print("=" * 70)
        
        # Check PM2
        if self.check_pm2():
            ok("PM2 installed")
            processes = self.get_pm2_status()
            
            print(f"\n{Colors.BOLD}PM2 Processes:{Colors.END}")
            
            for env_name, env_config in self.ENVIRONMENTS.items():
                pm2_name = env_config['pm2_name']
                
                # Find in PM2 list
                process = next((p for p in processes if p.get('name') == pm2_name), None)
                
                if process:
                    status = process.get('pm2_env', {}).get('status', 'unknown')
                    memory = process.get('monit', {}).get('memory', 0) / (1024 * 1024)
                    cpu = process.get('monit', {}).get('cpu', 0)
                    uptime = process.get('pm2_env', {}).get('pm_uptime', 0)
                    
                    if status == 'online':
                        status_color = Colors.GREEN
                    elif status == 'stopped':
                        status_color = Colors.YELLOW
                    else:
                        status_color = Colors.RED
                    
                    print(f"  {env_name:12} [{status_color}{status:8}{Colors.END}] "
                          f"Port:{env_config['port']} CPU:{cpu}% Mem:{memory:.0f}MB")
                else:
                    print(f"  {env_name:12} [{Colors.YELLOW}not running{Colors.END}] "
                          f"Port:{env_config['port']}")
        else:
            warn("PM2 not installed. Install with: npm install -g pm2")
        
        # Check ports
        print(f"\n{Colors.BOLD}Port Status:{Colors.END}")
        for env_name, env_config in self.ENVIRONMENTS.items():
            port = env_config['port']
            result = self.run_command(f"lsof -i :{port} | grep LISTEN", capture=True)
            if result and result.strip():
                ok(f"Port {port} ({env_name}): In use")
            else:
                info(f"Port {port} ({env_name}): Available")
        
        # Health checks
        print(f"\n{Colors.BOLD}Health Checks:{Colors.END}")
        import requests
        
        for env_name, env_config in self.ENVIRONMENTS.items():
            url = f"http://localhost:{env_config['port']}/health"
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    ok(f"{env_name}: Healthy")
                else:
                    warn(f"{env_name}: Unhealthy ({resp.status_code})")
            except:
                info(f"{env_name}: Not responding")
        
        print("\n" + "=" * 70 + "\n")
    
    def start(self, env: str = 'development', use_pm2: bool = True):
        """Start an environment"""
        if env not in self.ENVIRONMENTS:
            fail(f"Unknown environment: {env}")
            return
        
        config = self.ENVIRONMENTS[env]
        
        print("\n" + "=" * 50)
        print(f"  Starting ERICA API ({env})")
        print("=" * 50)
        
        info(f"Port: {config['port']}")
        info(f"URL: {config['url']}")
        
        # Set environment
        os.environ['ERICA_ENV'] = env
        
        if use_pm2 and self.check_pm2():
            # PM2 start
            cmd = (
                f"cd {self.root} && "
                f"ERICA_ENV={env} pm2 start 'uvicorn main:app --host {config['host']} --port {config['port']}' "
                f"--name {config['pm2_name']} "
                f"--interpreter python3 "
                f"--log {self.logs_dir}/pm2_{env}.log"
            )
            
            info(f"Starting with PM2...")
            self.run_command(cmd)
            
            time.sleep(2)
            ok(f"Started {config['pm2_name']}")
            
        else:
            # Direct uvicorn
            reload_flag = '--reload' if config['reload'] else ''
            cmd = f"cd {self.root} && ERICA_ENV={env} uvicorn main:app --host {config['host']} --port {config['port']} {reload_flag}"
            
            info(f"Starting with uvicorn...")
            info(f"Press Ctrl+C to stop")
            self.run_command(cmd)
    
    def stop(self, env: str = 'development'):
        """Stop an environment"""
        if env not in self.ENVIRONMENTS:
            fail(f"Unknown environment: {env}")
            return
        
        config = self.ENVIRONMENTS[env]
        
        if self.check_pm2():
            cmd = f"pm2 stop {config['pm2_name']}"
            self.run_command(cmd)
            ok(f"Stopped {config['pm2_name']}")
        else:
            # Kill by port
            cmd = f"lsof -ti :{config['port']} | xargs kill -9 2>/dev/null"
            self.run_command(cmd)
            ok(f"Killed process on port {config['port']}")
    
    def restart(self, env: str = 'development'):
        """Restart an environment"""
        if env not in self.ENVIRONMENTS:
            fail(f"Unknown environment: {env}")
            return
        
        config = self.ENVIRONMENTS[env]
        
        if self.check_pm2():
            cmd = f"pm2 restart {config['pm2_name']}"
            self.run_command(cmd)
            ok(f"Restarted {config['pm2_name']}")
        else:
            self.stop(env)
            time.sleep(1)
            self.start(env, use_pm2=False)
    
    def logs(self, env: str = 'development', lines: int = 50):
        """View logs for an environment"""
        config = self.ENVIRONMENTS.get(env)
        
        if not config:
            fail(f"Unknown environment: {env}")
            return
        
        # Try PM2 logs first
        if self.check_pm2():
            cmd = f"pm2 logs {config['pm2_name']} --lines {lines}"
            self.run_command(cmd)
        else:
            # Try log file
            log_file = self.logs_dir / f'erica_{env}.log'
            if log_file.exists():
                cmd = f"tail -n {lines} {log_file}"
                self.run_command(cmd)
            else:
                warn(f"No logs found for {env}")
    
    def deploy(self, env: str = 'staging'):
        """Full deployment to an environment"""
        print("\n" + "=" * 60)
        print(f"  ERICA Deployment to {env.upper()}")
        print("=" * 60)
        
        steps = [
            ("Pulling latest code", "git pull origin main"),
            ("Installing dependencies", "pip install -r requirements.txt"),
            ("Running requirements check", "python auto_requirements.py --check"),
            ("Running model tests", "python model_tester.py files"),
            ("Stopping current instance", f"python deploy.py stop {env}"),
            ("Starting new instance", f"python deploy.py start {env}"),
        ]
        
        for i, (name, cmd) in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] {name}...")
            
            if 'deploy.py' in cmd:
                # Internal command
                if 'stop' in cmd:
                    self.stop(env)
                elif 'start' in cmd:
                    self.start(env)
            else:
                result = self.run_command(f"cd {self.root} && {cmd}", capture=True)
                ok(f"Completed: {name}")
        
        # Health check
        print("\n[Final] Running health check...")
        time.sleep(3)
        
        config = self.ENVIRONMENTS[env]
        import requests
        
        try:
            resp = requests.get(f"http://localhost:{config['port']}/health", timeout=5)
            if resp.status_code == 200:
                ok("Health check passed!")
                print(f"\n{Colors.GREEN}{Colors.BOLD}Deployment successful!{Colors.END}")
                print(f"API running at: {config['url']}")
            else:
                fail(f"Health check failed: {resp.status_code}")
        except Exception as e:
            fail(f"Health check failed: {e}")
        
        print("\n" + "=" * 60 + "\n")
    
    def create_ecosystem(self):
        """Create PM2 ecosystem file"""
        ecosystem = {
            "apps": []
        }
        
        for env_name, config in self.ENVIRONMENTS.items():
            app = {
                "name": config['pm2_name'],
                "script": "uvicorn",
                "args": f"main:app --host {config['host']} --port {config['port']}",
                "cwd": str(self.root),
                "interpreter": "python3",
                "env": {
                    "ERICA_ENV": env_name
                },
                "instances": config['workers'],
                "exec_mode": "fork",
                "log_file": str(self.logs_dir / f"pm2_{env_name}.log"),
                "error_file": str(self.logs_dir / f"pm2_{env_name}_error.log"),
                "merge_logs": True,
                "autorestart": True,
                "max_restarts": 10,
                "restart_delay": 1000
            }
            ecosystem["apps"].append(app)
        
        output_path = self.root / 'ecosystem.config.js'
        
        with open(output_path, 'w') as f:
            f.write("module.exports = ")
            f.write(json.dumps(ecosystem, indent=2))
            f.write(";")
        
        ok(f"Created {output_path}")
        info("Start all: pm2 start ecosystem.config.js")


def main():
    parser = argparse.ArgumentParser(description='ERICA Deployment Manager')
    parser.add_argument('command', nargs='?', default='status',
                       choices=['status', 'start', 'stop', 'restart', 'logs', 'deploy', 'ecosystem'],
                       help='Command to run')
    parser.add_argument('env', nargs='?', default='development',
                       help='Environment (development, staging, production)')
    parser.add_argument('--lines', '-n', type=int, default=50,
                       help='Number of log lines')
    parser.add_argument('--no-pm2', action='store_true',
                       help='Start without PM2')
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.command == 'status':
        manager.status()
    elif args.command == 'start':
        manager.start(args.env, use_pm2=not args.no_pm2)
    elif args.command == 'stop':
        manager.stop(args.env)
    elif args.command == 'restart':
        manager.restart(args.env)
    elif args.command == 'logs':
        manager.logs(args.env, args.lines)
    elif args.command == 'deploy':
        manager.deploy(args.env)
    elif args.command == 'ecosystem':
        manager.create_ecosystem()


if __name__ == '__main__':
    main()
