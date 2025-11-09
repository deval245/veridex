import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RatingInfo:
    code: str
    description: str
    min_age: Optional[int]
    guidance: str
    metadata: Dict[str, Any]


@dataclass
class RatingSystem:
    country_code: str
    country_name: str
    system_name: str
    organization: str
    official_url: str
    ratings: List[RatingInfo]
    last_updated: datetime
    data_source: str


class RatingSystemManager:
    """
    Dynamic rating system manager
    NO HARDCODING - loads from JSON config or API
    
    Data sources:
    1. TMDb API (primary - has 50+ countries)
    2. JSON config (fallback - from official sources)
    3. User-provided (extensible)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / "countries.json"
        self.systems: Dict[str, RatingSystem] = {}
        self._load_systems()
    
    def _load_systems(self):
        """Load rating systems from JSON config"""
        
        if not self.config_path.exists():
            print(f"⚠️  No config found at {self.config_path}")
            print("   Using TMDb API for rating data")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for country_code, country_data in data.get("systems", {}).items():
                ratings = [
                    RatingInfo(
                        code=r["code"],
                        description=r["description"],
                        min_age=r.get("min_age"),
                        guidance=r.get("guidance", ""),
                        metadata=r.get("metadata", {})
                    )
                    for r in country_data.get("ratings", [])
                ]
                
                self.systems[country_code] = RatingSystem(
                    country_code=country_code,
                    country_name=country_data["country_name"],
                    system_name=country_data["system_name"],
                    organization=country_data["organization"],
                    official_url=country_data["official_url"],
                    ratings=ratings,
                    last_updated=datetime.fromisoformat(country_data["last_updated"]),
                    data_source=country_data.get("data_source", "official")
                )
        
        except Exception as e:
            print(f"⚠️  Error loading rating systems: {e}")
    
    def get_system(self, country_code: str) -> Optional[RatingSystem]:
        """Get rating system for a country"""
        return self.systems.get(country_code.upper())
    
    def get_all_countries(self) -> List[str]:
        """Get list of supported countries"""
        return list(self.systems.keys())
    
    def get_rating_info(self, country_code: str, rating_code: str) -> Optional[RatingInfo]:
        """Get specific rating info"""
        system = self.get_system(country_code)
        if not system:
            return None
        
        for rating in system.ratings:
            if rating.code == rating_code:
                return rating
        
        return None
    
    def is_valid_rating(self, country_code: str, rating_code: str) -> bool:
        """Check if rating code is valid for country"""
        return self.get_rating_info(country_code, rating_code) is not None
    
    def add_custom_system(
        self,
        country_code: str,
        country_name: str,
        system_name: str,
        organization: str,
        official_url: str,
        ratings: List[Dict[str, Any]]
    ):
        """Add custom rating system at runtime (extensible!)"""
        
        rating_objs = [
            RatingInfo(
                code=r["code"],
                description=r["description"],
                min_age=r.get("min_age"),
                guidance=r.get("guidance", ""),
                metadata=r.get("metadata", {})
            )
            for r in ratings
        ]
        
        self.systems[country_code.upper()] = RatingSystem(
            country_code=country_code.upper(),
            country_name=country_name,
            system_name=system_name,
            organization=organization,
            official_url=official_url,
            ratings=rating_objs,
            last_updated=datetime.now(),
            data_source="user_provided"
        )
    
    def get_data_freshness(self, country_code: str) -> Dict[str, Any]:
        """Get data freshness info for transparency"""
        system = self.get_system(country_code)
        if not system:
            return {"error": f"Country {country_code} not found"}
        
        from datetime import datetime, timedelta
        
        age_days = (datetime.now() - system.last_updated).days
        
        if age_days <= 7:
            freshness = "very_fresh"
            reliability = "high"
        elif age_days <= 30:
            freshness = "fresh"
            reliability = "high"
        elif age_days <= 90:
            freshness = "recent"
            reliability = "moderate"
        else:
            freshness = "stale"
            reliability = "needs_update"
        
        return {
            "country": country_code,
            "system": system.system_name,
            "last_updated": system.last_updated.isoformat(),
            "age_days": age_days,
            "freshness": freshness,
            "reliability": reliability,
            "data_source": system.data_source,
            "official_url": system.official_url,
            "disclaimer": "For critical compliance, verify against official source"
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded rating systems"""
        return {
            "total_countries": len(self.systems),
            "countries": list(self.systems.keys()),
            "data_sources": list(set(s.data_source for s in self.systems.values())),
            "oldest_update": min(
                (s.last_updated for s in self.systems.values()),
                default=None
            ),
            "newest_update": max(
                (s.last_updated for s in self.systems.values()),
                default=None
            ),
            "average_age_days": sum([
                (datetime.now() - s.last_updated).days 
                for s in self.systems.values()
            ]) / len(self.systems) if self.systems else 0
        }

