"""
SAFE DATA EXPANSION: Fetch Public TMDB Data for Disney Coverage
================================================================

This script fetches PUBLIC data from TMDB API for countries that Disney uses.
- Uses TMDB (public database) NOT Disney proprietary data
- No IP violation
- Safe for training

Priority countries to add:
1. Singapore (SG) - MDA system
2. Brazil (BR) - DJCTQ system
3. South Korea (KR) - KMRB system
4. Turkey (TR) - AI system
5. Taiwan (TW) - Custom APAC
6. Netherlands (NL) - Kijkwijzer
7. Hong Kong (HK) - Custom APAC
"""

import os
import json
import aiohttp
import asyncio
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

# Priority countries to fetch (based on Disney usage analysis)
PRIORITY_COUNTRIES = [
    'SG',  # Singapore - 2,018 Disney records
    'BR',  # Brazil - 2,005 Disney records
    'KR',  # South Korea - 1,524 Disney records
    'TR',  # Turkey - 1,413 Disney records
    'TW',  # Taiwan - 942 Disney records
    'NL',  # Netherlands - 904 Disney records
    'HK',  # Hong Kong - 846 Disney records
    'CH',  # Switzerland - 536 Disney records
    'NZ',  # New Zealand - 505 Disney records
    'ES',  # Spain - 81 Disney records (but important market)
    'IT',  # Italy - (important EU market)
    'MX',  # Mexico - (important LATAM market)
]

# Target samples per country
TARGET_PER_COUNTRY = 3000

API_KEY = os.getenv('TMDB_API_KEY', '300967e993b79558a6c09662675f7cd9')
BASE_URL = "https://api.themoviedb.org/3"


class SafeDataExpander:
    """
    Fetch PUBLIC TMDB data (NOT Disney data) for missing countries
    """
    
    def __init__(self):
        self.session = None
        self.movies = []
        self.stats = defaultdict(int)
        
    async def fetch_json(self, url):
        """Fetch with retry"""
        for attempt in range(3):
            try:
                async with self.session.get(url, timeout=10, ssl=False) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    await asyncio.sleep(1)
            except Exception as e:
                if attempt == 2:
                    print(f"❌ Failed: {url[:80]}")
                await asyncio.sleep(2)
        return None
    
    async def fetch_movie_details(self, movie_id):
        """Fetch movie details + release dates (for ratings)"""
        # Get basic details
        url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
        details = await self.fetch_json(url)
        
        if not details:
            return None
        
        # Get release dates (contains ratings per country)
        url = f"{BASE_URL}/movie/{movie_id}/release_dates?api_key={API_KEY}"
        release_data = await self.fetch_json(url)
        
        if not release_data:
            return details
        
        # Extract ratings by country
        ratings = {}
        for result in release_data.get('results', []):
            country = result.get('iso_3166_1')
            releases = result.get('release_dates', [])
            
            for release in releases:
                cert = release.get('certification', '').strip()
                if cert:
                    ratings[country] = cert
                    break
        
        details['ratings'] = ratings
        return details
    
    async def discover_movies_for_country(self, country, limit=3000):
        """Discover movies that have ratings in a specific country"""
        print(f"\n🔍 Fetching PUBLIC TMDB data for {country}...")
        
        collected = []
        page = 1
        max_pages = min(50, (limit // 20) + 1)
        
        while page <= max_pages and len(collected) < limit:
            # Discover popular movies
            url = f"{BASE_URL}/discover/movie"
            url += f"?api_key={API_KEY}"
            url += f"&page={page}"
            url += f"&sort_by=popularity.desc"
            url += f"&vote_count.gte=50"  # Quality filter
            url += f"&with_original_language=en|{country.lower()}"  # Prefer English or local
            
            data = await self.fetch_json(url)
            
            if not data or 'results' not in data:
                break
            
            # Fetch details for each movie
            tasks = []
            for movie in data['results']:
                if len(collected) >= limit:
                    break
                tasks.append(self.fetch_movie_details(movie['id']))
            
            results = await asyncio.gather(*tasks)
            
            for movie_data in results:
                if not movie_data:
                    continue
                
                # Only keep if it has the country we want
                ratings = movie_data.get('ratings', {})
                if country in ratings:
                    collected.append(movie_data)
                    self.stats[f'{country}_found'] += 1
            
            print(f"  Page {page}/{max_pages}: Found {len(collected)}/{limit} movies with {country} ratings")
            page += 1
            await asyncio.sleep(0.3)  # Rate limiting
        
        return collected
    
    async def expand_dataset(self):
        """Main expansion logic"""
        print("="*80)
        print("SAFE DATA EXPANSION: Fetching PUBLIC TMDB Data")
        print("="*80)
        print(f"Target countries: {PRIORITY_COUNTRIES}")
        print(f"Target per country: {TARGET_PER_COUNTRY}")
        print(f"Total target: {len(PRIORITY_COUNTRIES) * TARGET_PER_COUNTRY:,} samples")
        print("\n⚠️  Using PUBLIC TMDB data (NOT Disney proprietary data)")
        print("="*80)
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            all_movies = {}
            
            for country in PRIORITY_COUNTRIES:
                movies = await self.discover_movies_for_country(country, TARGET_PER_COUNTRY)
                
                # Merge into main collection
                for movie in movies:
                    movie_id = movie['id']
                    if movie_id not in all_movies:
                        # First time seeing this movie
                        all_movies[movie_id] = {
                            'id': movie_id,
                            'title': movie.get('title', ''),
                            'overview': movie.get('overview', ''),
                            'vote_count': movie.get('vote_count', 0),
                            'vote_average': movie.get('vote_average', 0),
                            'release_date': movie.get('release_date', ''),
                            'genres': [g['name'] for g in movie.get('genres', [])],
                            'ratings': movie.get('ratings', {}),
                            'poster_url': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path', '')}" if movie.get('poster_path') else None,
                            'backdrop_url': f"https://image.tmdb.org/t/p/w1280{movie.get('backdrop_path', '')}" if movie.get('backdrop_path') else None,
                        }
                    else:
                        # Movie already exists, merge ratings
                        all_movies[movie_id]['ratings'].update(movie.get('ratings', {}))
                
                print(f"✅ {country}: {len(movies)} movies collected")
                print(f"   Total unique movies so far: {len(all_movies):,}")
        
        return list(all_movies.values())
    
    def save_expanded_dataset(self, movies, output_path):
        """Save expanded dataset"""
        # Load existing 48K data
        existing_path = Path("data/multimodal_48k_final.json")
        if existing_path.exists():
            with open(existing_path) as f:
                existing_data = json.load(f)
            existing_movies = {m['id']: m for m in existing_data.get('movies', [])}
        else:
            existing_movies = {}
        
        # Merge new movies with existing
        for movie in movies:
            movie_id = movie['id']
            if movie_id in existing_movies:
                # Merge ratings
                existing_movies[movie_id]['ratings'].update(movie['ratings'])
            else:
                # New movie
                existing_movies[movie_id] = movie
        
        # Create final dataset
        final_movies = list(existing_movies.values())
        
        # Calculate statistics
        country_counts = Counter()
        rating_counts = Counter()
        for movie in final_movies:
            for country, rating in movie.get('ratings', {}).items():
                country_counts[country] += 1
                rating_counts[rating] += 1
        
        # Build metadata
        metadata = {
            'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_movies': len(final_movies),
            'version': 'expanded-for-disney-coverage',
            'countries': sorted(country_counts.keys()),
            'source': 'TMDB Public API (NO Disney proprietary data)',
            'expansion_info': {
                'original_countries': ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP'],
                'added_countries': PRIORITY_COUNTRIES,
                'country_counts': dict(country_counts.most_common()),
                'total_ratings': len(rating_counts),
            }
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
        
        print(f"\n" + "="*80)
        print(f"✅ EXPANDED DATASET SAVED")
        print(f"="*80)
        print(f"Output: {output_path}")
        print(f"Total movies: {len(final_movies):,}")
        print(f"Total countries: {len(country_counts)}")
        print(f"Original countries: 7")
        print(f"Added countries: {len(PRIORITY_COUNTRIES)}")
        print(f"Total countries now: {len(country_counts)}")
        print(f"\nTop 20 countries by sample count:")
        for country, count in country_counts.most_common(20):
            print(f"  {country}: {count:,} samples")
        print(f"\n⚠️  Dataset uses PUBLIC TMDB data (NO Disney IP)")
        print(f"="*80)


async def main():
    expander = SafeDataExpander()
    
    print("\n⚠️  IMPORTANT: IP COMPLIANCE")
    print("="*80)
    print("This script fetches PUBLIC data from TMDB API")
    print("It does NOT use Disney's proprietary PCON ratings")
    print("It is SAFE and legal to use for training")
    print("="*80)
    print("\n✅ Starting data collection (non-interactive mode)")
    
    # Fetch expanded data
    new_movies = await expander.expand_dataset()
    
    # Save
    output_path = "data/multimodal_expanded_disney_coverage.json"
    expander.save_expanded_dataset(new_movies, output_path)
    
    print(f"\n✅ Next steps:")
    print(f"1. Review the expanded dataset")
    print(f"2. Update your Colab script to use: multimodal_expanded_disney_coverage.json")
    print(f"3. Retrain the model")
    print(f"4. Expect 50-60% Disney PCON coverage (vs 11% now)")


if __name__ == '__main__':
    asyncio.run(main())

