from typing import Optional, Dict, Any, List
import aiohttp
from src.config import Settings


class TMDbAdapter:
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
    
    async def get_movie(self, movie_id: str) -> Dict[str, Any]:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}/movie/{movie_id}"
        params = {"api_key": self.api_key}
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_content_ratings(self, movie_id: str) -> List[Dict[str, str]]:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}/movie/{movie_id}/release_dates"
        params = {"api_key": self.api_key}
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            ratings = []
            for result in data.get("results", []):
                country = result.get("iso_3166_1")
                for release in result.get("release_dates", []):
                    certification = release.get("certification")
                    if certification:
                        ratings.append({
                            "country": country,
                            "rating": certification
                        })
            return ratings
    
    async def search_movie(self, query: str) -> List[Dict[str, Any]]:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}/search/movie"
        params = {"api_key": self.api_key, "query": query}
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("results", [])

