#!/usr/bin/env python3
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.tmdb import TMDbAdapter


class DatasetBuilder:
    
    ENDPOINTS = {
        "popular": "/movie/popular",
        "top_rated": "/movie/top_rated",
        "now_playing": "/movie/now_playing"
    }
    
    DEFAULT_REGIONS = ["US", "GB", "DE", "FR", "JP", "BR", "IN", "CA", "AU", "MX"]
    RATE_LIMIT_DELAY = 0.25
    MAX_PAGES_PER_ENDPOINT = {"popular": 25, "top_rated": 15, "now_playing": 10}
    
    def __init__(self, api_key: str, output_dir: Optional[Path] = None):
        self.api_key = api_key
        self.output_dir = output_dir or Path("data/dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def build(
        self,
        target_movies: int = 1000,
        regions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        
        regions = regions or self.DEFAULT_REGIONS
        
        print("=" * 80)
        print("BUILDING LARGE-SCALE DATASET")
        print("=" * 80)
        print(f"Target: {target_movies} movies | Regions: {len(regions)}")
        print()
        
        movies = []
        
        async with TMDbAdapter(self.api_key) as adapter:
            for endpoint_name, limit in [("popular", 500), ("top_rated", 300), ("now_playing", 200)]:
                print(f"📥 Fetching {endpoint_name}...")
                batch = await self._fetch_movies(adapter, endpoint_name, limit)
                movies.extend(batch)
                print(f"   ✓ {len(batch)} movies")
        
        unique_movies = {m.content_id: m for m in movies}.values()
        movies_list = list(unique_movies)
        
        print(f"\n✅ Total unique: {len(movies_list)}")
        
        stats = self._analyze(movies_list, regions)
        dataset_path = await self._save(movies_list, stats)
        
        print(f"\n💾 Saved: {dataset_path}\n")
        self._print_stats(stats)
        
        return {"movies": movies_list, "stats": stats, "path": dataset_path}
    
    async def _fetch_movies(
        self,
        adapter: TMDbAdapter,
        endpoint_name: str,
        limit: int
    ) -> List:
        
        movies = []
        max_pages = self.MAX_PAGES_PER_ENDPOINT.get(endpoint_name, 10)
        pages = min((limit // 20) + 1, max_pages)
        
        for page in range(1, pages + 1):
            try:
                url = f"{adapter.BASE_URL}{self.ENDPOINTS[endpoint_name]}"
                params = {"api_key": adapter.api_key, "page": page, "language": "en-US"}
                
                async with adapter.session.get(url, params=params) as response:
                    data = await response.json()
                    
                    for movie_data in data.get("results", []):
                        if len(movies) >= limit:
                            return movies
                        
                        try:
                            movie = await adapter.fetch_content_details(str(movie_data["id"]))
                            movies.append(movie)
                        except Exception:
                            continue
                
                await asyncio.sleep(self.RATE_LIMIT_DELAY)
                
            except Exception:
                continue
        
        return movies
    
    def _analyze(self, movies: List, regions: List[str]) -> Dict[str, Any]:
        
        ratings_by_region = {region: {} for region in regions}
        movies_with_ratings = {region: 0 for region in regions}
        
        for movie in movies:
            for region in regions:
                rating = movie.get_rating(region)
                if rating:
                    movies_with_ratings[region] += 1
                    ratings_by_region[region][rating] = ratings_by_region[region].get(rating, 0) + 1
        
        genre_counts = {}
        for movie in movies:
            for genre in movie.genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        year_counts = {}
        for movie in movies:
            if movie.release_date:
                year = movie.release_date.year
                year_counts[year] = year_counts.get(year, 0) + 1
        
        return {
            "total_movies": len(movies),
            "ratings_by_region": ratings_by_region,
            "movies_with_ratings": movies_with_ratings,
            "genre_counts": dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            "year_distribution": dict(sorted(year_counts.items(), reverse=True)[:10]),
            "total_validations_possible": sum(movies_with_ratings.values())
        }
    
    async def _save(self, movies: List, stats: Dict[str, Any]) -> Path:
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = self.output_dir / f"veridex_dataset_{timestamp}.json"
        
        movies_data = []
        for movie in movies:
            movies_data.append({
                "content_id": movie.content_id,
                "title": movie.title,
                "content_type": movie.content_type,
                "release_date": movie.release_date.isoformat() if movie.release_date else None,
                "regions": movie.regions,
                "genres": movie.genres,
                "ratings": movie.ratings,
                "metadata": movie.metadata
            })
        
        dataset = {
            "metadata": {
                "created_at": timestamp,
                "total_movies": len(movies),
                "version": "1.0",
                "source": "TMDb API"
            },
            "statistics": stats,
            "movies": movies_data
        }
        
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return dataset_path
    
    def _print_stats(self, stats: Dict[str, Any]):
        
        print("=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        print()
        
        print(f"📊 Total Movies: {stats['total_movies']}")
        print(f"📊 Total Validations: {stats['total_validations_possible']}")
        print()
        
        print("📈 Coverage by Region:")
        for region, count in sorted(stats['movies_with_ratings'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / stats['total_movies']) * 100
            print(f"   {region}: {count:4d} ({pct:5.1f}%)")
        print()
        
        print("📈 Top Genres:")
        for genre, count in list(stats['genre_counts'].items())[:10]:
            print(f"   {genre:20s}: {count:4d}")
        print()
        
        print("📈 US Rating Distribution:")
        us_ratings = stats['ratings_by_region'].get('US', {})
        for rating, count in sorted(us_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"   {rating:10s}: {count:4d}")
        print()


async def main():
    api_key = "300967e993b79558a6c09662675f7cd9"
    builder = DatasetBuilder(api_key)
    
    result = await builder.build(
        target_movies=1000,
        regions=["US", "GB", "DE", "FR", "JP", "BR", "IN", "CA", "AU", "MX"]
    )
    
    print("=" * 80)
    print("✅ COMPLETE")
    print(f"Movies: {len(result['movies'])} | Path: {result['path']}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
