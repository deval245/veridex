import json
import pprint

# Load data
with open('data/multimodal_expanded_coverage.json') as f:
    data = json.load(f)

# Check structure
movies = data['movies'] if isinstance(data, dict) and 'movies' in data else data

print("=" * 80)
print("DATA STRUCTURE CHECK")
print("=" * 80)
print(f"Total samples: {len(movies)}")
print(f"\nFirst movie:")
pprint.pprint(movies[0], depth=2, width=100)

print("\n" + "=" * 80)
print("RATINGS CHECK (First 10)")
print("=" * 80)
for i, m in enumerate(movies[:10]):
    title = m.get('title', 'N/A')[:40]
    ratings = m.get('ratings', 'NONE')
    country = m.get('country', 'NONE')
    print(f"{i+1:2d}. {title:40s} | Ratings: {ratings!r} | Country: {country!r}")

# Check if ratings have country-specific data
print("\n" + "=" * 80)
print("CHECKING IF RATINGS CONTAIN COUNTRY INFO")
print("=" * 80)

ratings_with_country = 0
for m in movies[:100]:
    ratings = m.get('ratings', {})
    if isinstance(ratings, dict) and len(ratings) > 0:
        print(f"\nMovie: {m.get('title', 'N/A')[:50]}")
        print(f"  Ratings structure: {ratings}")
        ratings_with_country += 1
        if ratings_with_country >= 3:
            break

