#!/usr/bin/env python3
"""
DOWNLOAD IMAGES FOR 48K DATASET
================================
Downloads all posters and backdrops from TMDb URLs
Creates images_48k.zip for Colab upload
"""

import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import zipfile
import time

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# CONFIG
DATASET = Path("data/multimodal_48k_final.json")
OUTPUT_DIR = Path("data/images_48k")
ZIP_OUTPUT = Path("data/images_48k.zip")
WORKERS = 20

# Stats
stats = {"posters": 0, "backdrops": 0, "errors": 0, "skipped": 0}
stats_lock = Lock()

def download_image(url, output_path):
    """Download single image"""
    if output_path.exists():
        with stats_lock:
            stats["skipped"] += 1
        return True
    
    try:
        r = requests.get(url, timeout=30, verify=False)
        if r.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    
    with stats_lock:
        stats["errors"] += 1
    return False

def main():
    print("=" * 70)
    print("📥 DOWNLOADING IMAGES FOR 48K DATASET")
    print("=" * 70)
    
    # Load dataset
    with open(DATASET) as f:
        data = json.load(f)
    
    movies = data.get("movies", [])
    print(f"Movies: {len(movies):,}")
    
    # Collect all image URLs
    poster_dir = OUTPUT_DIR / "posters"
    backdrop_dir = OUTPUT_DIR / "backdrops"
    poster_dir.mkdir(parents=True, exist_ok=True)
    backdrop_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for movie in movies:
        movie_id = movie["id"]
        
        # Poster
        if movie.get("poster_url"):
            tasks.append(("poster", movie["poster_url"], poster_dir / f"{movie_id}.jpg"))
        
        # Backdrop
        if movie.get("backdrop_url"):
            tasks.append(("backdrop", movie["backdrop_url"], backdrop_dir / f"{movie_id}.jpg"))
    
    print(f"Total images to download: {len(tasks):,}")
    print(f"Workers: {WORKERS}")
    print("-" * 70)
    
    # Download with thread pool
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_image, url, path): (type_, url, path) for type_, url, path in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            type_, url, path = futures[future]
            success = future.result()
            
            if success:
                with stats_lock:
                    if type_ == "poster":
                        stats["posters"] += 1
                    else:
                        stats["backdrops"] += 1
            
            # Progress every 100 images
            if i % 100 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(tasks) - i) / rate / 60
                print(f"Progress: {i:,}/{len(tasks):,} | "
                      f"Posters: {stats['posters']:,} | "
                      f"Backdrops: {stats['backdrops']:,} | "
                      f"Errors: {stats['errors']:,} | "
                      f"ETA: {eta:.0f}m")
    
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Posters: {stats['posters']:,}")
    print(f"Backdrops: {stats['backdrops']:,}")
    print(f"Errors: {stats['errors']:,}")
    print(f"Skipped (already exists): {stats['skipped']:,}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print("=" * 70)
    
    # Create zip
    print("\n📦 Creating ZIP file...")
    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add posters
        for img in poster_dir.glob("*.jpg"):
            zf.write(img, f"images/posters/{img.name}")
        
        # Add backdrops
        for img in backdrop_dir.glob("*.jpg"):
            zf.write(img, f"images/backdrops/{img.name}")
    
    zip_size = ZIP_OUTPUT.stat().st_size / (1024 ** 3)
    print(f"✅ Created: {ZIP_OUTPUT}")
    print(f"✅ Size: {zip_size:.2f} GB")
    print("\n" + "=" * 70)
    print("✅ READY FOR COLAB!")
    print("=" * 70)
    print("Upload these 2 files to /content/ in Colab:")
    print(f"1. {DATASET}")
    print(f"2. {ZIP_OUTPUT}")
    print("=" * 70)

if __name__ == "__main__":
    main()









