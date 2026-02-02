#!/usr/bin/env python3
"""
Version Manager - Semantic versioning and release management

Features:
- Bump version (major, minor, patch)
- Create git tags
- Generate changelog
- Update version in files

Usage:
    python version_manager.py show             # Show current version
    python version_manager.py bump patch       # 2.0.0 -> 2.0.1
    python version_manager.py bump minor       # 2.0.0 -> 2.1.0
    python version_manager.py bump major       # 2.0.0 -> 3.0.0
    python version_manager.py tag              # Create git tag
    python version_manager.py changelog        # Generate changelog
"""

import os
import re
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def ok(msg): print(f"{Colors.GREEN}✓ {msg}{Colors.END}")
def fail(msg): print(f"{Colors.RED}✗ {msg}{Colors.END}")
def info(msg): print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


class VersionManager:
    """Manage semantic versioning for ERICA API"""
    
    VERSION_FILE = 'VERSION'
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.version_file = self.root / self.VERSION_FILE
        
    def get_version(self) -> str:
        """Get current version"""
        # Try VERSION file
        if self.version_file.exists():
            return self.version_file.read_text().strip()
        
        # Try config.py
        config_file = self.root / 'config.py'
        if config_file.exists():
            content = config_file.read_text()
            match = re.search(r"ERICA_VERSION['\"]?,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                return match.group(1)
        
        # Try .env files
        for env_file in self.root.glob('.env*'):
            content = env_file.read_text()
            match = re.search(r'ERICA_VERSION=([^\n]+)', content)
            if match:
                return match.group(1).strip()
        
        return '2.0.0'  # Default
    
    def set_version(self, version: str):
        """Set version in all relevant files"""
        # Update VERSION file
        self.version_file.write_text(version + '\n')
        ok(f"Updated {self.VERSION_FILE}")
        
        # Update .env files
        for env_file in self.root.glob('.env*'):
            content = env_file.read_text()
            new_content = re.sub(
                r'ERICA_VERSION=.*',
                f'ERICA_VERSION={version}',
                content
            )
            env_file.write_text(new_content)
            ok(f"Updated {env_file.name}")
    
    def parse_version(self, version: str) -> Tuple[int, int, int, str]:
        """Parse version string into components"""
        # Handle versions like 2.0.0-dev, 2.0.0-rc.1
        match = re.match(r'(\d+)\.(\d+)\.(\d+)(?:-(.+))?', version)
        if not match:
            raise ValueError(f"Invalid version format: {version}")
        
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))
        suffix = match.group(4) or ''
        
        return major, minor, patch, suffix
    
    def bump(self, bump_type: str, suffix: str = '') -> str:
        """Bump version"""
        current = self.get_version()
        major, minor, patch, old_suffix = self.parse_version(current)
        
        if bump_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif bump_type == 'minor':
            minor += 1
            patch = 0
        elif bump_type == 'patch':
            patch += 1
        elif bump_type == 'release':
            # Remove suffix for release
            suffix = ''
        
        new_version = f"{major}.{minor}.{patch}"
        if suffix:
            new_version += f"-{suffix}"
        
        return new_version
    
    def show(self):
        """Show current version"""
        version = self.get_version()
        
        print("\n" + "=" * 50)
        print("  ERICA Version Information")
        print("=" * 50)
        print(f"\n  Current Version: {Colors.BOLD}{version}{Colors.END}")
        
        # Parse components
        try:
            major, minor, patch, suffix = self.parse_version(version)
            print(f"\n  Major: {major}")
            print(f"  Minor: {minor}")
            print(f"  Patch: {patch}")
            if suffix:
                print(f"  Suffix: {suffix}")
        except ValueError as e:
            print(f"  {Colors.RED}{e}{Colors.END}")
        
        # Show git info
        try:
            tag = subprocess.run(
                ['git', 'describe', '--tags', '--always'],
                capture_output=True, text=True, cwd=self.root
            )
            if tag.returncode == 0:
                print(f"\n  Git tag: {tag.stdout.strip()}")
            
            commit = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, cwd=self.root
            )
            if commit.returncode == 0:
                print(f"  Git commit: {commit.stdout.strip()}")
        except:
            pass
        
        print("\n" + "=" * 50 + "\n")
    
    def do_bump(self, bump_type: str, suffix: str = ''):
        """Perform version bump"""
        current = self.get_version()
        new_version = self.bump(bump_type, suffix)
        
        print("\n" + "=" * 50)
        print("  Version Bump")
        print("=" * 50)
        print(f"\n  Current: {current}")
        print(f"  New:     {Colors.GREEN}{new_version}{Colors.END}")
        
        confirm = input("\n  Proceed? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("  Cancelled")
            return
        
        self.set_version(new_version)
        
        print(f"\n{Colors.GREEN}Version bumped to {new_version}{Colors.END}")
        print("\n" + "=" * 50 + "\n")
    
    def tag(self, message: str = None):
        """Create git tag for current version"""
        version = self.get_version()
        tag_name = f"v{version}"
        
        if not message:
            message = f"Release {version}"
        
        print("\n" + "=" * 50)
        print("  Create Git Tag")
        print("=" * 50)
        print(f"\n  Tag: {tag_name}")
        print(f"  Message: {message}")
        
        confirm = input("\n  Create tag? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("  Cancelled")
            return
        
        try:
            # Create tag
            subprocess.run(
                ['git', 'tag', '-a', tag_name, '-m', message],
                check=True, cwd=self.root
            )
            ok(f"Created tag: {tag_name}")
            
            push = input("  Push tag to remote? [y/N]: ").strip().lower()
            if push == 'y':
                subprocess.run(
                    ['git', 'push', 'origin', tag_name],
                    check=True, cwd=self.root
                )
                ok(f"Pushed tag to remote")
                
        except subprocess.CalledProcessError as e:
            fail(f"Failed to create tag: {e}")
        
        print("\n" + "=" * 50 + "\n")
    
    def changelog(self, since_tag: str = None):
        """Generate changelog from git commits"""
        print("\n" + "=" * 50)
        print("  Changelog Generator")
        print("=" * 50)
        
        try:
            # Get commits since last tag
            if since_tag:
                cmd = ['git', 'log', f'{since_tag}..HEAD', '--oneline']
            else:
                # Get last tag
                last_tag = subprocess.run(
                    ['git', 'describe', '--tags', '--abbrev=0'],
                    capture_output=True, text=True, cwd=self.root
                )
                if last_tag.returncode == 0:
                    tag = last_tag.stdout.strip()
                    cmd = ['git', 'log', f'{tag}..HEAD', '--oneline']
                    print(f"\n  Since tag: {tag}")
                else:
                    # All commits
                    cmd = ['git', 'log', '--oneline', '-20']
                    print("\n  Last 20 commits:")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root)
            
            if result.returncode == 0 and result.stdout:
                commits = result.stdout.strip().split('\n')
                
                print(f"\n  Changes ({len(commits)} commits):")
                
                features = []
                fixes = []
                other = []
                
                for commit in commits:
                    if any(word in commit.lower() for word in ['feat', 'add', 'new']):
                        features.append(commit)
                    elif any(word in commit.lower() for word in ['fix', 'bug', 'patch']):
                        fixes.append(commit)
                    else:
                        other.append(commit)
                
                if features:
                    print(f"\n  {Colors.GREEN}Features:{Colors.END}")
                    for c in features:
                        print(f"    • {c}")
                
                if fixes:
                    print(f"\n  {Colors.YELLOW}Fixes:{Colors.END}")
                    for c in fixes:
                        print(f"    • {c}")
                
                if other:
                    print(f"\n  {Colors.BLUE}Other:{Colors.END}")
                    for c in other[:10]:  # Limit
                        print(f"    • {c}")
                    if len(other) > 10:
                        print(f"    ... and {len(other) - 10} more")
                
                # Offer to save
                save = input("\n  Save to CHANGELOG.md? [y/N]: ").strip().lower()
                if save == 'y':
                    self._save_changelog(features, fixes, other)
            else:
                info("No commits found")
                
        except Exception as e:
            fail(f"Error generating changelog: {e}")
        
        print("\n" + "=" * 50 + "\n")
    
    def _save_changelog(self, features, fixes, other):
        """Save changelog to file"""
        version = self.get_version()
        date = datetime.now().strftime('%Y-%m-%d')
        
        changelog_path = self.root / 'CHANGELOG.md'
        
        # Read existing
        existing = ''
        if changelog_path.exists():
            existing = changelog_path.read_text()
        
        # Generate new entry
        entry = f"\n## [{version}] - {date}\n\n"
        
        if features:
            entry += "### Features\n"
            for c in features:
                entry += f"- {c}\n"
            entry += "\n"
        
        if fixes:
            entry += "### Fixes\n"
            for c in fixes:
                entry += f"- {c}\n"
            entry += "\n"
        
        if other:
            entry += "### Other\n"
            for c in other[:10]:
                entry += f"- {c}\n"
            entry += "\n"
        
        # Prepend to file
        if existing:
            # Insert after header
            if '# Changelog' in existing:
                parts = existing.split('\n', 2)
                new_content = f"{parts[0]}\n{entry}{parts[2] if len(parts) > 2 else ''}"
            else:
                new_content = f"# Changelog\n{entry}\n{existing}"
        else:
            new_content = f"# Changelog\n{entry}"
        
        changelog_path.write_text(new_content)
        ok(f"Updated CHANGELOG.md")


def main():
    parser = argparse.ArgumentParser(description='ERICA Version Manager')
    parser.add_argument('command', nargs='?', default='show',
                       choices=['show', 'bump', 'tag', 'changelog'],
                       help='Command to run')
    parser.add_argument('type', nargs='?', default='patch',
                       choices=['major', 'minor', 'patch', 'release'],
                       help='Bump type')
    parser.add_argument('--suffix', '-s', default='',
                       help='Version suffix (e.g., dev, rc.1)')
    parser.add_argument('--message', '-m',
                       help='Tag message')
    parser.add_argument('--since', 
                       help='Generate changelog since this tag')
    
    args = parser.parse_args()
    
    manager = VersionManager()
    
    if args.command == 'show':
        manager.show()
    elif args.command == 'bump':
        manager.do_bump(args.type, args.suffix)
    elif args.command == 'tag':
        manager.tag(args.message)
    elif args.command == 'changelog':
        manager.changelog(args.since)


if __name__ == '__main__':
    main()
