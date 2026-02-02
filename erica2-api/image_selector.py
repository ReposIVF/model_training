#!/usr/bin/env python3
"""
Image Selector - Interactive tool to select and process local images

Features:
- Browse local images
- Run local ranking
- Upload to cloud for processing
- Compare results
- Export rankings

Usage:
    python image_selector.py                    # Interactive mode
    python image_selector.py --folder ./images  # Process folder
    python image_selector.py --image img.jpg    # Process single image
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import config, get_api_key, get_validation_key
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False


class ImageSelector:
    """Interactive image selection and ranking tool"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.temp_dir = self.root / 'temp_images'
        self.results_dir = self.root / 'ranking_results'
        self.selected_images: List[Path] = []
        
    def setup(self):
        """Create necessary directories"""
        self.temp_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
    def find_images(self, folder: Path) -> List[Path]:
        """Find all images in a folder"""
        images = []
        for ext in self.SUPPORTED_FORMATS:
            images.extend(folder.glob(f'*{ext}'))
            images.extend(folder.glob(f'*{ext.upper()}'))
        return sorted(images)
    
    def display_images(self, images: List[Path]) -> None:
        """Display list of images with numbers"""
        print("\n" + "=" * 50)
        print("  Available Images")
        print("=" * 50)
        
        for i, img in enumerate(images, 1):
            size = img.stat().st_size / 1024
            print(f"  [{i:2d}] {img.name} ({size:.1f} KB)")
        
        print("=" * 50)
        print(f"  Total: {len(images)} images")
        print("=" * 50 + "\n")
    
    def select_images(self, images: List[Path]) -> List[Path]:
        """Interactive image selection"""
        self.display_images(images)
        
        print("Selection options:")
        print("  - Enter numbers separated by commas (e.g., 1,3,5)")
        print("  - Enter range with dash (e.g., 1-5)")
        print("  - Enter 'all' to select all")
        print("  - Enter 'q' to quit")
        
        selection = input("\nSelect images: ").strip().lower()
        
        if selection == 'q':
            return []
        
        if selection == 'all':
            return images
        
        selected = []
        try:
            # Parse selection
            parts = selection.replace(' ', '').split(',')
            for part in parts:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    for i in range(start, end + 1):
                        if 1 <= i <= len(images):
                            selected.append(images[i - 1])
                else:
                    idx = int(part)
                    if 1 <= idx <= len(images):
                        selected.append(images[idx - 1])
        except ValueError:
            print("Invalid selection format")
            return self.select_images(images)
        
        return selected
    
    def copy_to_temp(self, images: List[Path], session_id: str) -> List[Dict]:
        """Copy selected images to temp folder for processing"""
        session_dir = self.temp_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        embryos = []
        for i, img in enumerate(images):
            # Copy image
            dest = session_dir / f"embryo_{i+1}{img.suffix}"
            shutil.copy(img, dest)
            
            embryos.append({
                'embryo': f'local_{i+1}',
                'image': img.name,
                'local_path': str(dest),
                'original_path': str(img),
                'isEmbryo': True,
                'pgt': ''
            })
        
        return embryos
    
    def run_local_ranking(self, embryos: List[Dict], mother_age: int = 35) -> List[Dict]:
        """Run local ranking on embryos"""
        try:
            from utils.erica_cropper import cropper
            from utils.erica_pipeline import erica_pipeline
            
            print("\n[1/4] Running cropper...")
            embryos = cropper(embryos, {})
            
            if not embryos:
                print("No embryos detected after cropping")
                return []
            
            print(f"  Detected {len(embryos)} embryos")
            
            print("\n[2/4] Running full pipeline...")
            ranked = erica_pipeline(
                embryos_list=embryos,
                mother_age=mother_age,
                oocyte_origin='Autologous',
                models={}
            )
            
            print(f"\n[3/4] Ranking complete!")
            return ranked
            
        except Exception as e:
            print(f"Error during ranking: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def display_results(self, results: List[Dict]) -> None:
        """Display ranking results"""
        print("\n" + "=" * 60)
        print("  RANKING RESULTS")
        print("=" * 60)
        
        for i, emb in enumerate(results, 1):
            score = emb.get('score', 'N/A')
            letter = emb.get('letter', '?')
            embryo_id = emb.get('embryo', 'unknown')
            
            print(f"\n  Rank #{i} ({letter})")
            print(f"    Score: {score}")
            print(f"    Embryo: {embryo_id}")
            
            # Show additional info if available
            if 'is_embryo' in emb:
                print(f"    Is Embryo: {emb['is_embryo']}")
        
        print("\n" + "=" * 60)
    
    def save_results(self, results: List[Dict], session_id: str) -> Path:
        """Save results to JSON file"""
        output_file = self.results_dir / f"ranking_{session_id}.json"
        
        output = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'total_embryos': len(results),
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        return output_file
    
    def upload_to_cloud(self, images: List[Path], object_id: str) -> Dict:
        """Upload images and trigger cloud ranking"""
        import requests
        
        if not CONFIG_LOADED:
            print("Config not loaded, cannot upload to cloud")
            return {'error': 'Config not loaded'}
        
        api_key = get_api_key()
        val_key = get_validation_key()
        
        # For now, trigger ranking on existing cycle
        # (Images should already be in S3 for the cycle)
        
        url = f"{config.api_url}/rankthisone"
        
        try:
            response = requests.post(
                url,
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
            
            return response.json()
            
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup(self, session_id: str):
        """Clean up temp files"""
        session_dir = self.temp_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"Cleaned up temp files for session {session_id}")
    
    def interactive_mode(self):
        """Run interactive mode"""
        self.setup()
        
        print("\n" + "=" * 60)
        print("  ERICA Image Selector - Interactive Mode")
        print("=" * 60)
        
        # Get folder
        default_folder = self.root / 'test_images'
        folder_input = input(f"\nEnter folder path [{default_folder}]: ").strip()
        folder = Path(folder_input) if folder_input else default_folder
        
        if not folder.exists():
            print(f"Folder not found: {folder}")
            create = input("Create folder? [y/n]: ").strip().lower()
            if create == 'y':
                folder.mkdir(parents=True)
                print(f"Created folder: {folder}")
                print("Add images to the folder and run again.")
                return
            return
        
        # Find images
        images = self.find_images(folder)
        
        if not images:
            print(f"No images found in {folder}")
            return
        
        # Select images
        selected = self.select_images(images)
        
        if not selected:
            print("No images selected")
            return
        
        print(f"\nSelected {len(selected)} images")
        
        # Get mother age
        age_input = input("Enter mother's age [35]: ").strip()
        mother_age = int(age_input) if age_input else 35
        
        # Create session
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"\nSession ID: {session_id}")
        
        # Copy and process
        print("\nPreparing images...")
        embryos = self.copy_to_temp(selected, session_id)
        
        # Run ranking
        print("\nRunning local ranking...")
        results = self.run_local_ranking(embryos, mother_age)
        
        if results:
            self.display_results(results)
            self.save_results(results, session_id)
        
        # Cleanup option
        cleanup = input("\nClean up temp files? [y/n]: ").strip().lower()
        if cleanup == 'y':
            self.cleanup(session_id)
    
    def process_folder(self, folder_path: str, mother_age: int = 35):
        """Process all images in a folder"""
        self.setup()
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            return
        
        images = self.find_images(folder)
        if not images:
            print(f"No images found in {folder}")
            return
        
        print(f"Found {len(images)} images")
        
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        embryos = self.copy_to_temp(images, session_id)
        results = self.run_local_ranking(embryos, mother_age)
        
        if results:
            self.display_results(results)
            self.save_results(results, session_id)
    
    def process_single(self, image_path: str, mother_age: int = 35):
        """Process a single image"""
        self.setup()
        
        image = Path(image_path)
        if not image.exists():
            print(f"Image not found: {image}")
            return
        
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        embryos = self.copy_to_temp([image], session_id)
        results = self.run_local_ranking(embryos, mother_age)
        
        if results:
            self.display_results(results)
            self.save_results(results, session_id)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ERICA Image Selector')
    parser.add_argument('--folder', '-f', help='Process all images in folder')
    parser.add_argument('--image', '-i', help='Process single image')
    parser.add_argument('--age', '-a', type=int, default=35, help='Mother age (default: 35)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    selector = ImageSelector()
    
    if args.folder:
        selector.process_folder(args.folder, args.age)
    elif args.image:
        selector.process_single(args.image, args.age)
    else:
        selector.interactive_mode()


if __name__ == '__main__':
    main()
