"""
ASYNC DATA EXPANSION with LIVE MONITORING
=========================================
Fetch PUBLIC TMDB data for Disney coverage with real-time progress tracking
"""

import os
import json
import aiohttp
import asyncio
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
import sys

# Priority countries
PRIORITY_COUNTRIES = [
    ('SG', 'Singapore', 2000),
    ('BR', 'Brazil', 2000),
    ('KR', 'South Korea', 1500),
    ('TR', 'Turkey', 1400),
    ('TW', 'Taiwan', 1000),
    ('NL', 'Netherlands', 1000),
    ('HK', 'Hong Kong', 900),
    ('CH', 'Switzerland', 600),
    ('NZ', 'New Zealand', 600),
    ('ES', 'Spain', 800),
    ('IT', 'Italy', 800),
    ('MX', 'Mexico', 800),
]

API_KEY = '300967e993b79558a6c09662675f7cd9'
BASE_URL = "https://api.themoviedb.org/3"


class AsyncDataExpander:
    def __init__(self):
        self.session = None
        self.progress = defaultdict(int)
        self.errors = defaultdict(int)
        self.total_fetched = 0
        self.start_time = None
        
    def print_progress(self, country_code, found, target, status=""):
        """Print progress with colors"""
        elapsed = (datetime.now() - self.start_time).seconds if self.start_time else 0
        pct = (found / target * 100) if target > 0 else 0
        bar_len = 30
        filled = int(bar_len * found / target) if target > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        
        print(f"\r  {country_code}: [{bar}] {found:4d}/{target} ({pct:5.1f}%) {status}", 
              end='', flush=True)
    
    async def fetch_json(self, url):
        """Fetch with retry and timeout"""
        for attempt in range(3):
            try:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10), ssl=False) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:  # Rate limit
                        await asyncio.sleep(2)
                    else:
                        await asyncio.sleep(0.5)
            except asyncio.TimeoutError:
                if attempt == 2:
                    self.errors['timeout'] += 1
                await asyncio.sleep(1)
            except Exception as e:
                if attempt == 2:
                    self.errors['other'] += 1
                await asyncio.sleep(1)
        return None
    
    async def fetch_movie_with_ratings(self, movie_id):
        """Fetch movie details + ratings"""
        # Get basic details
        url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
        details = await self.fetch_json(url)
        
        if not details or 'overview' not in details:
            return None
        
        # Get release dates (contains ratings)
        url = f"{BASE_URL}/movie/{movie_id}/release_dates?api_key={API_KEY}"
        release_data = await self.fetch_json(url)
        
        # Extract ratings
        ratings = {}
        if release_data:
            for result in release_data.get('results', []):
                country = result.get('iso_3166_1', '')
                for release in result.get('release_dates', []):
                    cert = release.get('certification', '').strip()
                    if cert:
                        ratings[country] = cert
                        break
        
        if not ratings:
            return None
        
        return {
            'id': details['id'],
            'title': details.get('title', ''),
            'overview': details.get('overview', ''),
            'vote_count': details.get('vote_count', 0),
            'vote_average': details.get('vote_average', 0),
            'release_date': details.get('release_date', ''),
            'genres': [g['name'] for g in details.get('genres', [])] if 'genres' in details else [],
            'ratings': ratings,
            'poster_path': details.get('poster_path'),
            'backdrop_path': details.get('backdrop_path'),
        }
    
    async def discover_for_country(self, country_code, country_name, target):
        """Discover movies for a specific country"""
        collected = []
        page = 1
        max_pages = 100
        
        self.print_progress(country_code, 0, target, "Starting...")
        
        while page <= max_pages and len(collected) < target:
            # Discover movies
            url = f"{BASE_URL}/discover/movie"
            url += f"?api_key={API_KEY}"
            url += f"&page={page}"
            url += f"&sort_by=popularity.desc"
            url += f"&vote_count.gte=30"
            
            data = await self.fetch_json(url)
            
            if not data or 'results' not in data:
                break
            
            # Fetch details in batches
            tasks = []
            for movie in data['results'][:10]:  # Process 10 per page
                if len(collected) >= target:
                    break
                tasks.append(self.fetch_movie_with_ratings(movie['id']))
            
            results = await asyncio.gather(*tasks)
            
            # Filter for target country
            for movie_data in results:
                if not movie_data:
                    continue
                
                if country_code in movie_data['ratings']:
                    collected.append(movie_data)
                    self.total_fetched += 1
                    self.print_progress(country_code, len(collected), target, 
                                      f"Total: {self.total_fetched}")
            
            page += 1
            await asyncio.sleep(0.25)  # Rate limiting
        
        self.print_progress(country_code, len(collected), target, "✓ Done")
        print()  # New line
        
        return country_code, collected
    
    async def expand_all(self):
        """Fetch all countries in parallel"""
        print("="*80)
        print("ASYNC DATA EXPANSION - LIVE MONITORING")
        print("="*80)
        print(f"Target countries: {len(PRIORITY_COUNTRIES)}")
        print(f"Total target samples: {sum(t for _, _, t in PRIORITY_COUNTRIES):,}")
        print(f"⚠️  Using PUBLIC TMDB data (NO Disney IP)")
        print("="*80)
        print()
        
        self.start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Create tasks for all countries (parallel!)
            tasks = [
                self.discover_for_country(code, name, target)
                for code, name, target in PRIORITY_COUNTRIES
            ]
            
            # Run all in parallel
            results = await asyncio.gather(*tasks)
            
            # Organize results
            all_movies = {}
            country_stats = {}
            
            for country_code, movies in results:
                country_stats[country_code] = len(movies)
                
                for movie in movies:
                    movie_id = movie['id']
                    if movie_id not in all_movies:
                        all_movies[movie_id] = movie
                    else:
                        # Merge ratings
                        all_movies[movie_id]['ratings'].update(movie['ratings'])
            
            print()
            print("="*80)
            print("COLLECTION SUMMARY")
            print("="*80)
            for country_code, count in sorted(country_stats.items()):
                print(f"  {country_code}: {count:5,} samples")
            print(f"\nTotal unique movies: {len(all_movies):,}")
            print(f"Total samples: {self.total_fetched:,}")
            print(f"Errors: {sum(self.errors.values())}")
            
            elapsed = (datetime.now() - self.start_time).seconds
            print(f"Time: {elapsed//60}m {elapsed%60}s")
            print("="*80)
            
            return list(all_movies.values())
    
    def merge_and_save(self, new_movies, output_path):
        """Merge with existing 48K data and save"""
        print("\n📊 Merging with existing dataset...")
        
        # Load existing
        existing_path = Path("data/multimodal_48k_final.json")
        if existing_path.exists():
            with open(existing_path) as f:
                existing_data = json.load(f)
            existing_movies = {m['id']: m for m in existing_data.get('movies', [])}
            print(f"  Loaded {len(existing_movies):,} existing movies")
        else:
            existing_movies = {}
            print(f"  No existing data found")
        
        # Merge
        for movie in new_movies:
            movie_id = movie['id']
            if movie_id in existing_movies:
                existing_movies[movie_id]['ratings'].update(movie['ratings'])
            else:
                existing_movies[movie_id] = movie
        
        final_movies = list(existing_movies.values())
        
        # Calculate stats
        country_counts = Counter()
        for movie in final_movies:
            for country in movie.get('ratings', {}).keys():
                country_counts[country] += 1
        
        # Build metadata
        metadata = {
            'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_movies': len(final_movies),
            'version': 'expanded-disney-coverage',
            'countries': sorted(country_counts.keys()),
            'source': 'TMDB Public API',
            'country_counts': dict(country_counts.most_common()),
        }
        
        output_data = {
            'metadata': metadata,
            'movies': final_movies
        }
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ SAVED: {output_path}")
        print(f"  Total movies: {len(final_movies):,}")
        print(f"  Total countries: {len(country_counts)}")
        print(f"\nTop 20 countries:")
        for country, count in country_counts.most_common(20):
            print(f"  {country}: {count:,}")
        print()


async def main():
    expander = AsyncDataExpander()
    
    # Fetch data (all countries in parallel)
    new_movies = await expander.expand_all()
    
    # Merge and save
    output_path = "data/multimodal_expanded_disney_coverage.json"
    expander.merge_and_save(new_movies, output_path)
    
    print("="*80)
    print("✅ DATA EXPANSION COMPLETE!")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Ready for training!")
    print("="*80)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)








