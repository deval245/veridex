import asyncio
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.veridex import VERIDEX
from dotenv import load_dotenv

load_dotenv()


async def main():
    tmdb_key = os.getenv("TMDB_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not tmdb_key or not openai_key:
        print("Error: TMDB_API_KEY and OPENAI_API_KEY must be set in .env file")
        return
    
    veridex = VERIDEX(tmdb_api_key=tmdb_key)
    
    print("🔍 VERIDEX Content Rating Validation Demo\n")
    
    test_cases = [
        {"movie_id": "550", "expected_rating": "R", "title": "Fight Club"},
        {"movie_id": "157336", "expected_rating": "PG-13", "title": "Interstellar"},
        {"movie_id": "27205", "expected_rating": "PG-13", "title": "Inception"},
    ]
    
    for test in test_cases:
        print(f"Validating: {test['title']} (ID: {test['movie_id']})")
        print(f"  Expected Rating: {test['expected_rating']}")
        
        result = await veridex.validate_content_rating(
            movie_id=test["movie_id"],
            expected_rating=test["expected_rating"],
            country="US"
        )
        
        print(f"  Actual Rating: {result.actual}")
        print(f"  Status: {result.status}")
        print(f"  Reason: {result.reason}")
        print(f"  Confidence: {result.confidence:.2f}\n")
    
    await veridex.cleanup()
    print("✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())

