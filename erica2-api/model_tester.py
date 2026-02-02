#!/usr/bin/env python3
"""
Model Tester - Comprehensive testing for ERICA ML models

Features:
- Model loading tests
- Inference benchmarks
- Memory usage analysis
- Batch processing tests
- Model comparison

Usage:
    python model_tester.py test          # Run all tests
    python model_tester.py benchmark     # Run benchmarks
    python model_tester.py memory        # Check memory usage
    python model_tester.py compare       # Compare model versions
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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
def warn(msg): print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")
def info(msg): print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


class ModelTester:
    """Test ERICA ML models"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.models_dir = self.root / 'models'
        self.results: Dict[str, any] = {}
        
    def get_model_paths(self) -> Dict[str, Path]:
        """Get paths to all models"""
        return {
            'cropper': self.models_dir / 'erica_cropper.pt',
            'segmentor': self.models_dir / 'erica_segmentor_n.pt',
            'scoring': self.models_dir / 'erica_model2.pth',
            'scaler': self.models_dir / 'scaler_info.json',
        }
    
    def check_files(self) -> bool:
        """Check if all model files exist"""
        print("\n" + "=" * 50)
        print("  Model Files Check")
        print("=" * 50 + "\n")
        
        all_exist = True
        models = self.get_model_paths()
        
        for name, path in models.items():
            if path.exists():
                size = path.stat().st_size / (1024 * 1024)
                ok(f"{name}: {path.name} ({size:.2f} MB)")
            else:
                fail(f"{name}: {path.name} NOT FOUND")
                all_exist = False
        
        return all_exist
    
    def test_cropper(self) -> Tuple[bool, float]:
        """Test cropper model loading"""
        print("\n" + "-" * 40)
        print("Testing Cropper Model (YOLO)")
        print("-" * 40)
        
        try:
            from ultralytics import YOLO
            
            model_path = self.models_dir / 'erica_cropper.pt'
            if not model_path.exists():
                fail("Model file not found")
                return False, 0
            
            start = time.time()
            model = YOLO(str(model_path))
            load_time = time.time() - start
            
            ok(f"Model loaded in {load_time:.2f}s")
            
            # Test inference with dummy image
            import numpy as np
            from PIL import Image
            
            dummy_img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            
            start = time.time()
            results = model.predict(dummy_img, verbose=False)
            inference_time = time.time() - start
            
            ok(f"Inference completed in {inference_time * 1000:.1f}ms")
            
            self.results['cropper'] = {
                'status': 'pass',
                'load_time': load_time,
                'inference_time': inference_time
            }
            
            return True, load_time
            
        except Exception as e:
            fail(f"Cropper test failed: {e}")
            self.results['cropper'] = {'status': 'fail', 'error': str(e)}
            return False, 0
    
    def test_segmentor(self) -> Tuple[bool, float]:
        """Test segmentor model loading"""
        print("\n" + "-" * 40)
        print("Testing Segmentor Model (YOLO)")
        print("-" * 40)
        
        try:
            from ultralytics import YOLO
            
            model_path = self.models_dir / 'erica_segmentor_n.pt'
            if not model_path.exists():
                fail("Model file not found")
                return False, 0
            
            start = time.time()
            model = YOLO(str(model_path))
            load_time = time.time() - start
            
            ok(f"Model loaded in {load_time:.2f}s")
            
            # Test inference
            import numpy as np
            from PIL import Image
            
            dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            
            start = time.time()
            results = model.predict(dummy_img, verbose=False)
            inference_time = time.time() - start
            
            ok(f"Inference completed in {inference_time * 1000:.1f}ms")
            
            self.results['segmentor'] = {
                'status': 'pass',
                'load_time': load_time,
                'inference_time': inference_time
            }
            
            return True, load_time
            
        except Exception as e:
            fail(f"Segmentor test failed: {e}")
            self.results['segmentor'] = {'status': 'fail', 'error': str(e)}
            return False, 0
    
    def test_scoring(self) -> Tuple[bool, float]:
        """Test scoring model loading"""
        print("\n" + "-" * 40)
        print("Testing Scoring Model (PyTorch)")
        print("-" * 40)
        
        try:
            import torch
            
            model_path = self.models_dir / 'erica_model2.pth'
            if not model_path.exists():
                fail("Model file not found")
                return False, 0
            
            start = time.time()
            
            # Load model state
            state = torch.load(model_path, map_location='cpu')
            load_time = time.time() - start
            
            ok(f"Model loaded in {load_time:.2f}s")
            
            # Check model structure
            if isinstance(state, dict):
                if 'state_dict' in state:
                    info(f"  Keys in state_dict: {len(state['state_dict'])}")
                elif 'model' in state:
                    info(f"  Model key found")
                else:
                    info(f"  Top-level keys: {list(state.keys())[:5]}")
            
            self.results['scoring'] = {
                'status': 'pass',
                'load_time': load_time
            }
            
            return True, load_time
            
        except Exception as e:
            fail(f"Scoring model test failed: {e}")
            self.results['scoring'] = {'status': 'fail', 'error': str(e)}
            return False, 0
    
    def test_scaler(self) -> Tuple[bool, float]:
        """Test scaler info loading"""
        print("\n" + "-" * 40)
        print("Testing Scaler Info")
        print("-" * 40)
        
        try:
            scaler_path = self.models_dir / 'scaler_info.json'
            if not scaler_path.exists():
                fail("Scaler file not found")
                return False, 0
            
            start = time.time()
            with open(scaler_path, 'r') as f:
                scaler_info = json.load(f)
            load_time = time.time() - start
            
            ok(f"Scaler loaded in {load_time * 1000:.2f}ms")
            
            # Check structure
            if 'mean' in scaler_info:
                info(f"  Features (mean): {len(scaler_info['mean'])}")
            if 'scale' in scaler_info:
                info(f"  Features (scale): {len(scaler_info['scale'])}")
            
            self.results['scaler'] = {
                'status': 'pass',
                'load_time': load_time
            }
            
            return True, load_time
            
        except Exception as e:
            fail(f"Scaler test failed: {e}")
            self.results['scaler'] = {'status': 'fail', 'error': str(e)}
            return False, 0
    
    def run_all_tests(self) -> Dict:
        """Run all model tests"""
        print("\n" + "=" * 60)
        print("  ERICA Model Tester - Full Test Suite")
        print("=" * 60)
        
        self.check_files()
        
        tests = [
            ('Cropper', self.test_cropper),
            ('Segmentor', self.test_segmentor),
            ('Scoring', self.test_scoring),
            ('Scaler', self.test_scaler),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            try:
                success, _ = test_func()
                if success:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                fail(f"{name}: Unexpected error - {e}")
                failed += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("  Test Summary")
        print("=" * 60)
        print(f"  Passed: {Colors.GREEN}{passed}{Colors.END}")
        print(f"  Failed: {Colors.RED}{failed}{Colors.END}")
        print("=" * 60 + "\n")
        
        self.results['summary'] = {
            'passed': passed,
            'failed': failed,
            'total': passed + failed,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results
    
    def benchmark(self, iterations: int = 10) -> Dict:
        """Run benchmark tests"""
        print("\n" + "=" * 60)
        print(f"  ERICA Model Benchmarks ({iterations} iterations)")
        print("=" * 60)
        
        import numpy as np
        from PIL import Image
        
        benchmarks = {}
        
        # Benchmark cropper
        print("\n[1/3] Benchmarking Cropper...")
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.models_dir / 'erica_cropper.pt'))
            
            img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            
            times = []
            for i in range(iterations):
                start = time.time()
                _ = model.predict(img, verbose=False)
                times.append(time.time() - start)
            
            benchmarks['cropper'] = {
                'mean_ms': np.mean(times) * 1000,
                'std_ms': np.std(times) * 1000,
                'min_ms': np.min(times) * 1000,
                'max_ms': np.max(times) * 1000
            }
            ok(f"Cropper: {benchmarks['cropper']['mean_ms']:.1f}ms ± {benchmarks['cropper']['std_ms']:.1f}ms")
            
        except Exception as e:
            fail(f"Cropper benchmark failed: {e}")
        
        # Benchmark segmentor
        print("\n[2/3] Benchmarking Segmentor...")
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.models_dir / 'erica_segmentor_n.pt'))
            
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            
            times = []
            for i in range(iterations):
                start = time.time()
                _ = model.predict(img, verbose=False)
                times.append(time.time() - start)
            
            benchmarks['segmentor'] = {
                'mean_ms': np.mean(times) * 1000,
                'std_ms': np.std(times) * 1000,
                'min_ms': np.min(times) * 1000,
                'max_ms': np.max(times) * 1000
            }
            ok(f"Segmentor: {benchmarks['segmentor']['mean_ms']:.1f}ms ± {benchmarks['segmentor']['std_ms']:.1f}ms")
            
        except Exception as e:
            fail(f"Segmentor benchmark failed: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("  Benchmark Summary")
        print("=" * 60)
        for model_name, stats in benchmarks.items():
            print(f"  {model_name}: {stats['mean_ms']:.1f}ms (±{stats['std_ms']:.1f}ms)")
        print("=" * 60 + "\n")
        
        return benchmarks
    
    def check_memory(self) -> Dict:
        """Check memory usage of models"""
        print("\n" + "=" * 60)
        print("  Memory Usage Analysis")
        print("=" * 60)
        
        try:
            import torch
            import psutil
        except ImportError:
            warn("psutil not installed. Run: pip install psutil")
            return {}
        
        process = psutil.Process()
        memory_info = {}
        
        # Baseline
        baseline = process.memory_info().rss / (1024 * 1024)
        info(f"Baseline memory: {baseline:.1f} MB")
        
        # Load cropper
        print("\nLoading cropper...")
        from ultralytics import YOLO
        cropper = YOLO(str(self.models_dir / 'erica_cropper.pt'))
        after_cropper = process.memory_info().rss / (1024 * 1024)
        memory_info['cropper'] = after_cropper - baseline
        ok(f"Cropper: +{memory_info['cropper']:.1f} MB")
        
        # Load segmentor
        print("\nLoading segmentor...")
        segmentor = YOLO(str(self.models_dir / 'erica_segmentor_n.pt'))
        after_segmentor = process.memory_info().rss / (1024 * 1024)
        memory_info['segmentor'] = after_segmentor - after_cropper
        ok(f"Segmentor: +{memory_info['segmentor']:.1f} MB")
        
        # Load scoring
        print("\nLoading scoring model...")
        scoring = torch.load(self.models_dir / 'erica_model2.pth', map_location='cpu')
        after_scoring = process.memory_info().rss / (1024 * 1024)
        memory_info['scoring'] = after_scoring - after_segmentor
        ok(f"Scoring: +{memory_info['scoring']:.1f} MB")
        
        # Total
        total = after_scoring - baseline
        memory_info['total'] = total
        
        print("\n" + "=" * 60)
        print("  Memory Summary")
        print("=" * 60)
        print(f"  Total model memory: {total:.1f} MB")
        print(f"  System total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"  System available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print("=" * 60 + "\n")
        
        return memory_info
    
    def save_results(self, results: Dict, filename: str = 'test_results.json'):
        """Save test results to file"""
        output_path = self.root / 'logs' / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ERICA Model Tester')
    parser.add_argument('command', nargs='?', default='test',
                       choices=['test', 'benchmark', 'memory', 'files'],
                       help='Command to run')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                       help='Benchmark iterations')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    tester = ModelTester()
    
    if args.command == 'test':
        results = tester.run_all_tests()
    elif args.command == 'benchmark':
        results = tester.benchmark(args.iterations)
    elif args.command == 'memory':
        results = tester.check_memory()
    elif args.command == 'files':
        tester.check_files()
        return
    
    if args.save:
        tester.save_results(results)


if __name__ == '__main__':
    main()
