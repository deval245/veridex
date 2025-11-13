import asyncio
import aiohttp
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
import aiofiles

TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

MAJOR_COUNTRIES = ['US', 'GB', 'DE', 'FR', 'JP', 'AU', 'CA']

async def download_image(session: aiohttp.ClientSession, url: str, filepath: Path) -> bool:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                content = await resp.read()
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(content)
                return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

async def fetch_movie_with_images(
    session: aiohttp.ClientSession,
    movie_id: int,
    image_dir: Path
) -> Dict[str, Any]:
    try:
        async with session.get(
            f"{BASE_URL}/movie/{movie_id}",
            params={'api_key': TMDB_API_KEY, 'append_to_response': 'release_dates,images'},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                return None
            
            movie = await resp.json()
            
            overview = movie.get('overview', '')
            if not overview or len(overview) < 30:
                return None
            
            ratings = {}
            if 'release_dates' in movie:
                for release in movie['release_dates'].get('results', []):
                    country = release.get('iso_3166_1')
                    if country not in MAJOR_COUNTRIES:
                        continue
                    
                    for date_info in release.get('release_dates', []):
                        cert = date_info.get('certification', '').strip()
                        if cert:
                            ratings[country] = cert
                            break
            
            if len(ratings) < 2:
                return None
            
            poster_path = None
            backdrop_path = None
            
            if movie.get('poster_path'):
                poster_url = f"{IMAGE_BASE_URL}{movie['poster_path']}"
                poster_filename = f"poster_{movie_id}.jpg"
                poster_filepath = image_dir / "posters" / poster_filename
                
                if await download_image(session, poster_url, poster_filepath):
                    poster_path = str(poster_filepath.relative_to(image_dir.parent))
            
            if movie.get('backdrop_path'):
                backdrop_url = f"{IMAGE_BASE_URL}{movie['backdrop_path']}"
                backdrop_filename = f"backdrop_{movie_id}.jpg"
                backdrop_filepath = image_dir / "backdrops" / backdrop_filename
                
                if await download_image(session, backdrop_url, backdrop_filepath):
                    backdrop_path = str(backdrop_filepath.relative_to(image_dir.parent))
            
            if not poster_path:
                return None
            
            genres = [g['name'] for g in movie.get('genres', [])]
            
            return {
                'id': movie['id'],
                'title': movie.get('title'),
                'ratings': ratings,
                'metadata': {
                    'overview': overview,
                    'genres': genres,
                    'release_date': movie.get('release_date'),
                    'vote_average': movie.get('vote_average'),
                    'popularity': movie.get('popularity')
                },
                'images': {
                    'poster': poster_path,
                    'backdrop': backdrop_path
                }
            }
    
    except Exception as e:
        print(f"Error fetching movie {movie_id}: {e}")
        return None

async def fetch_popular_movies(
    session: aiohttp.ClientSession,
    max_pages: int = 500
) -> List[int]:
    movie_ids = []
    
    for page in range(1, max_pages + 1):
        try:
            async with session.get(
                f"{BASE_URL}/movie/popular",
                params={'api_key': TMDB_API_KEY, 'page': page},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    movie_ids.extend([m['id'] for m in data.get('results', [])])
                
                if page % 10 == 0:
                    print(f"Fetched {len(movie_ids)} movie IDs from {page} pages...")
        
        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue
    
    return movie_ids

async def main():
    if not TMDB_API_KEY:
        print("Error: TMDB_API_KEY not set!")
        return
    
    target_movies = 10000
    print(f"🎬 Fetching multi-modal dataset (target: {target_movies} movies)")
    
    project_root = Path(__file__).parent.parent
    image_dir = project_root / "data" / "images"
    (image_dir / "posters").mkdir(parents=True, exist_ok=True)
    (image_dir / "backdrops").mkdir(parents=True, exist_ok=True)
    
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        print("🔍 Fetching movie IDs from TMDb...")
        movie_ids = await fetch_popular_movies(session, max_pages=500)
        print(f"✅ Found {len(movie_ids)} movie IDs")
        
        print(f"\n📥 Downloading movies with images...")
        movies = []
        
        tasks = [fetch_movie_with_images(session, mid, image_dir) for mid in movie_ids]
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching"):
            movie = await coro
            if movie:
                movies.append(movie)
            
            if len(movies) >= target_movies:
                break
        
        print(f"\n✅ Successfully fetched {len(movies)} movies with images")
        
        stats = {
            'total_movies': len(movies),
            'ratings_by_country': {},
            'with_poster': sum(1 for m in movies if m['images']['poster']),
            'with_backdrop': sum(1 for m in movies if m['images']['backdrop'])
        }
        
        for movie in movies:
            for country, rating in movie['ratings'].items():
                if country not in stats['ratings_by_country']:
                    stats['ratings_by_country'][country] = {}
                if rating not in stats['ratings_by_country'][country]:
                    stats['ratings_by_country'][country][rating] = 0
                stats['ratings_by_country'][country][rating] += 1
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = project_root / "data" / "dataset" / f"multimodal_dataset_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps({
                'metadata': {
                    'created_at': timestamp,
                    'total_movies': len(movies),
                    'version': '2.0-multimodal',
                    'source': 'TMDb API',
                    'countries': MAJOR_COUNTRIES
                },
                'statistics': stats,
                'movies': movies
            }, indent=2))
        
        print(f"\n💾 Dataset saved: {output_file}")
        print(f"\n📊 Statistics:")
        print(f"  Total movies: {len(movies):,}")
        print(f"  With poster: {stats['with_poster']:,}")
        print(f"  With backdrop: {stats['with_backdrop']:,}")
        print(f"\n🌍 Ratings by country:")
        for country, ratings in stats['ratings_by_country'].items():
            total = sum(ratings.values())
            print(f"  {country}: {total:,} movies, {len(ratings)} unique ratings")

if __name__ == '__main__':
    asyncio.run(main())









