from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ContentRecord:
    """Universal content record (works for ANY OTT platform)"""
    content_id: str
    title: str
    content_type: str  # "movie", "tv", "series"
    release_date: Optional[datetime]
    regions: List[str]
    genres: List[str]
    ratings: Dict[str, str]  # {region: rating}
    metadata: Dict[str, Any]
    
    def get_rating(self, region: str) -> Optional[str]:
        return self.ratings.get(region)


class ContentAdapter(ABC):
    """Base adapter for fetching content from any source"""
    
    @abstractmethod
    async def fetch_content(
        self,
        content_ids: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[ContentRecord]:
        """Fetch content records from source"""
        pass
    
    @abstractmethod
    async def fetch_content_details(self, content_id: str) -> ContentRecord:
        """Fetch detailed info for single content item"""
        pass
    
    def normalize_rating(self, rating: str, region: str) -> str:
        """Normalize rating to standard format"""
        rating_map = {
            "US": {
                "G": "G", "PG": "PG", "PG-13": "PG-13", 
                "R": "R", "NC-17": "NC-17", "NR": "NR"
            },
            "GB": {
                "U": "U", "PG": "PG", "12": "12", "12A": "12A",
                "15": "15", "18": "18", "R18": "R18"
            },
            "DE": {
                "0": "0", "6": "6", "12": "12", "16": "16", "18": "18"
            }
        }
        
        region_map = rating_map.get(region, {})
        return region_map.get(rating, rating)










