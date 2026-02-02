#!/usr/bin/env python3
"""
Auto Requirements - Automatically sync requirements.txt with installed packages

This script:
1. Scans all Python files for imports
2. Compares with installed packages
3. Updates requirements.txt automatically
4. Runs on API startup (optional)

Usage:
    python auto_requirements.py           # Analyze and update
    python auto_requirements.py --check   # Check only, don't update
    python auto_requirements.py --scan    # Show all imports found
"""

import os
import sys
import ast
import subprocess
from pathlib import Path
from typing import Set, Dict, List, Tuple
from datetime import datetime


# Standard library modules (Python 3.9+)
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect',
    'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd',
    'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
    'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg',
    'cProfile', 'crypt', 'csv', 'ctypes', 'curses', 'dataclasses', 'datetime',
    'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email',
    'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass',
    'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac',
    'html', 'http', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io',
    'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
    'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
    'mimetypes', 'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
    'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats',
    'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random',
    're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
    'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
    'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
    'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
    'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading',
    'time', 'timeit', 'tkinter', 'token', 'tokenize', 'trace', 'traceback',
    'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing', 'unicodedata',
    'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref',
    'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
    'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread'
}

# Map import names to package names (PyPI)
IMPORT_TO_PACKAGE = {
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'dotenv': 'python-dotenv',
    'jwt': 'PyJWT',
    'bs4': 'beautifulsoup4',
    'lxml': 'lxml',
    'dateutil': 'python-dateutil',
}

# Local modules to ignore
LOCAL_MODULES = {
    'config', 'erica_api', 'utils', 'models',
    'erica_cropper', 'erica_model', 'erica_pipeline', 'erica_img_process',
    'get_embryos_db', 'set_rank_db', 'download_img_s3', 'clean_files',
    'model_explorer', 'auto_requirements', 'dev_cli', 'image_selector',
    'model_tester', 'deploy', 'version_manager', 'logger'
}


class ImportScanner:
    """Scan Python files for imports"""
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.imports: Set[str] = set()
        self.files_scanned = 0
    
    def scan(self) -> Set[str]:
        """Scan all Python files and extract imports"""
        for py_file in self.root.rglob('*.py'):
            # Skip __pycache__ and venv
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
            
            self._scan_file(py_file)
        
        return self.imports
    
    def _scan_file(self, filepath: Path):
        """Extract imports from a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            self.files_scanned += 1
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_import(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_import(node.module)
                        
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  [WARN] Could not parse {filepath.name}: {e}")
    
    def _add_import(self, module_name: str):
        """Add a module to imports (base module only)"""
        base = module_name.split('.')[0]
        
        # Skip stdlib and local modules
        if base in STDLIB_MODULES or base in LOCAL_MODULES:
            return
        
        self.imports.add(base)


class RequirementsManager:
    """Manage requirements.txt"""
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.req_file = root_path / 'requirements.txt'
        self.req_file_alt = root_path / 'requeriments.txt'  # Handle typo
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages"""
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True
        )
        
        packages = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                name, version = line.split('==')
                packages[name.lower()] = version
            elif line.startswith('-e'):
                # Editable installs
                continue
        
        return packages
    
    def read_requirements(self) -> Dict[str, str]:
        """Read current requirements.txt"""
        req_file = self.req_file if self.req_file.exists() else self.req_file_alt
        
        if not req_file.exists():
            return {}
        
        packages = {}
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '==' in line:
                    name, version = line.split('==')
                    packages[name.lower()] = version
                elif '>=' in line:
                    name = line.split('>=')[0]
                    packages[name.lower()] = line.split('>=')[1]
                else:
                    packages[line.lower()] = ''
        
        return packages
    
    def update_requirements(self, imports: Set[str], installed: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """Update requirements.txt with new packages"""
        
        # Convert import names to package names
        required_packages = set()
        for imp in imports:
            pkg_name = IMPORT_TO_PACKAGE.get(imp, imp)
            required_packages.add(pkg_name.lower())
        
        # Find packages that are both imported and installed
        matched = []
        missing = []
        
        for pkg in sorted(required_packages):
            if pkg in installed:
                matched.append(f"{pkg}=={installed[pkg]}")
            else:
                # Check case-insensitive
                found = False
                for installed_pkg, version in installed.items():
                    if installed_pkg.lower() == pkg.lower():
                        matched.append(f"{installed_pkg}=={version}")
                        found = True
                        break
                if not found:
                    missing.append(pkg)
        
        # Write requirements.txt
        with open(self.req_file, 'w') as f:
            f.write(f"# ERICA API Requirements\n")
            f.write(f"# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"# Run: pip install -r requirements.txt\n\n")
            
            f.write("# Core Framework\n")
            core = ['fastapi', 'uvicorn', 'python-dotenv', 'requests']
            for pkg in core:
                version = installed.get(pkg, '')
                if version:
                    f.write(f"{pkg}=={version}\n")
                elif pkg.lower() in [m.lower() for m in matched]:
                    for m in matched:
                        if m.lower().startswith(pkg.lower()):
                            f.write(f"{m}\n")
                            break
            
            f.write("\n# ML/AI Libraries\n")
            ml = ['torch', 'torchvision', 'ultralytics', 'numpy', 'pandas', 'scikit-learn', 'pillow', 'opencv-python']
            for pkg in ml:
                pkg_lower = pkg.lower()
                for m in matched:
                    if m.lower().startswith(pkg_lower):
                        f.write(f"{m}\n")
                        break
            
            f.write("\n# Other Dependencies\n")
            written = set(core + ml)
            for m in sorted(matched):
                pkg_name = m.split('==')[0].lower()
                if pkg_name not in [w.lower() for w in written]:
                    f.write(f"{m}\n")
        
        return matched, missing


def analyze_and_report(check_only: bool = False, scan_only: bool = False):
    """Main analysis function"""
    root = Path(__file__).parent
    
    print("\n" + "=" * 60)
    print("  ERICA Auto-Requirements Scanner")
    print("=" * 60)
    
    # Scan imports
    print("\n[1/3] Scanning Python files for imports...")
    scanner = ImportScanner(root)
    imports = scanner.scan()
    print(f"  Scanned {scanner.files_scanned} files")
    print(f"  Found {len(imports)} unique external imports")
    
    if scan_only:
        print("\n  Imports found:")
        for imp in sorted(imports):
            pkg = IMPORT_TO_PACKAGE.get(imp, imp)
            if imp != pkg:
                print(f"    {imp} -> {pkg}")
            else:
                print(f"    {imp}")
        return
    
    # Get installed packages
    print("\n[2/3] Getting installed packages...")
    manager = RequirementsManager(root)
    installed = manager.get_installed_packages()
    print(f"  Found {len(installed)} installed packages")
    
    # Current requirements
    current = manager.read_requirements()
    print(f"  Current requirements.txt has {len(current)} packages")
    
    if check_only:
        print("\n  Analysis (check mode - no changes):")
        
        # Find missing
        for imp in sorted(imports):
            pkg = IMPORT_TO_PACKAGE.get(imp, imp).lower()
            if pkg in installed:
                print(f"    ✓ {imp}")
            else:
                print(f"    ✗ {imp} (not installed)")
        return
    
    # Update requirements
    print("\n[3/3] Updating requirements.txt...")
    matched, missing = manager.update_requirements(imports, installed)
    
    print(f"\n  Updated requirements.txt with {len(matched)} packages")
    
    if missing:
        print(f"\n  Missing packages (install manually):")
        for pkg in missing:
            print(f"    pip install {pkg}")
    
    print("\n" + "=" * 60)
    print("  Done! Run: pip install -r requirements.txt")
    print("=" * 60 + "\n")


def export_conda_environment():
    """Export conda environment if using miniconda/anaconda"""
    root = Path(__file__).parent
    
    # Check if running in conda environment
    conda_prefix = os.getenv('CONDA_PREFIX')
    conda_default_env = os.getenv('CONDA_DEFAULT_ENV')
    
    if not conda_prefix:
        print("[INFO] Not running in a conda environment")
        return False
    
    print(f"[INFO] Detected conda environment: {conda_default_env}")
    
    try:
        # Export full environment
        output_file = root / 'miniconda_requirements.yml'
        cmd = f"conda env export --from-history > {output_file}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[INFO] Exported conda environment to miniconda_requirements.yml")
            
            # Also create a simple txt version
            output_txt = root / 'miniconda_requirements.txt'
            cmd_txt = f"conda list --export > {output_txt}"
            subprocess.run(cmd_txt, shell=True, capture_output=True, text=True)
            print(f"[INFO] Exported package list to miniconda_requirements.txt")
            
            return True
        else:
            print(f"[WARN] Failed to export conda environment: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[WARN] Error exporting conda environment: {e}")
        return False


def startup_requirements_check():
    """Quick check on startup - called from main.py"""
    root = Path(__file__).parent
    
    # Export conda environment if available
    export_conda_environment()
    
    scanner = ImportScanner(root)
    imports = scanner.scan()
    
    manager = RequirementsManager(root)
    installed = manager.get_installed_packages()
    
    missing = []
    for imp in imports:
        pkg = IMPORT_TO_PACKAGE.get(imp, imp).lower()
        if pkg not in installed and pkg not in [i.lower() for i in installed]:
            missing.append(imp)
    
    if missing:
        print(f"[WARN] Missing packages: {', '.join(missing)}")
        print("[WARN] Run: python auto_requirements.py")
    else:
        print("[INFO] All imports satisfied")
    
    return len(missing) == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-sync requirements.txt')
    parser.add_argument('--check', action='store_true', help='Check only, no updates')
    parser.add_argument('--scan', action='store_true', help='Show all imports found')
    parser.add_argument('--conda', action='store_true', help='Export conda environment')
    
    args = parser.parse_args()
    
    if args.conda:
        export_conda_environment()
    else:
        analyze_and_report(check_only=args.check, scan_only=args.scan)
