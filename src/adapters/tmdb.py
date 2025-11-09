import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.adapters.base import ContentAdapter, ContentRecord


class TMDbAdapter(ContentAdapter):
    """
    Adapter for The Movie Database (TMDb) - Public API
    Works for ANY OTT platform (Netflix, Disney+, Hulu, etc.)
    """
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_content(
        self,
        content_ids: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[ContentRecord]:
        """Fetch popular movies from TMDb"""
        
        if content_ids:
            records = []
            for content_id in content_ids[:limit]:
                try:
                    record = await self.fetch_content_details(content_id)
                    records.append(record)
                except Exception as e:
                    print(f"Error fetching {content_id}: {e}")
            return records
        
        records = []
        pages = (limit // 20) + 1
        
        for page in range(1, min(pages + 1, 6)):
            url = f"{self.BASE_URL}/movie/popular"
            params = {
                "api_key": self.api_key,
                "page": page,
                "language": "en-US"
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                for movie in data.get("results", [])[:limit]:
                    try:
                        record = await self._movie_to_record(movie)
                        records.append(record)
                        
                        if len(records) >= limit:
                            return records
                    except Exception as e:
                        print(f"Error processing movie {movie.get('id')}: {e}")
        
        return records
    
    async def fetch_content_details(self, content_id: str) -> ContentRecord:
        """Fetch detailed movie info including ratings"""
        
        url = f"{self.BASE_URL}/movie/{content_id}"
        params = {
            "api_key": self.api_key,
            "append_to_response": "release_dates,content_ratings"
        }
        
        async with self.session.get(url, params=params) as response:
            movie = await response.json()
            return await self._movie_to_record(movie, detailed=True)
    
    async def _movie_to_record(
        self,
        movie: Dict[str, Any],
        detailed: bool = False
    ) -> ContentRecord:
        """Convert TMDb movie data to ContentRecord"""
        
        ratings = {}
        
        if detailed and "release_dates" in movie:
            for release in movie["release_dates"].get("results", []):
                country = release.get("iso_3166_1")
                for date_info in release.get("release_dates", []):
                    certification = date_info.get("certification")
                    if certification:
                        ratings[country] = self.normalize_rating(certification, country)
                        break
        
        release_date = None
        if movie.get("release_date"):
            try:
                release_date = datetime.strptime(movie["release_date"], "%Y-%m-%d")
            except:
                pass
        
        genres = []
        if "genres" in movie and movie["genres"]:
            genres = [g.get("name") for g in movie["genres"]]
        elif "genre_ids" in movie:
            genre_map = {28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 
                        80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
                        14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
                        9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
                        10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"}
            genres = [genre_map.get(gid, f"Genre-{gid}") for gid in movie.get("genre_ids", [])]
        
        return ContentRecord(
            content_id=str(movie["id"]),
            title=movie.get("title", "Unknown"),
            content_type="movie",
            release_date=release_date,
            regions=list(ratings.keys()) if ratings else ["US"],
            genres=genres,
            ratings=ratings,
            metadata={
                "overview": movie.get("overview"),
                "popularity": movie.get("popularity"),
                "vote_average": movie.get("vote_average"),
                "vote_count": movie.get("vote_count"),
                "adult": movie.get("adult", False),
                "original_language": movie.get("original_language")
            }
        )
    
    async def fetch_batch(
        self,
        start_id: int,
        count: int,
        delay: float = 0.25
    ) -> List[ContentRecord]:
        """Fetch a batch of movies by ID range (with rate limiting)"""
        
        records = []
        
        for movie_id in range(start_id, start_id + count):
            try:
                record = await self.fetch_content_details(str(movie_id))
                records.append(record)
                await asyncio.sleep(delay)
            except Exception as e:
                continue
        
        return records

