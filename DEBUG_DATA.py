"""
VERIDEX Data Quality Diagnostic
Checks label distribution, encoding, and data integrity
"""
import json
from collections import Counter
from pathlib import Path

def diagnose_data(json_path):
    """Run comprehensive data quality checks"""
    
    print("=" * 80)
    print("VERIDEX DATA DIAGNOSTIC")
    print("=" * 80)
    
    # Load data
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"\n✓ Loaded {len(data)} samples")
    
    # Check rating structure
    print("\n" + "=" * 80)
    print("RATING STRUCTURE CHECK")
    print("=" * 80)
    
    rating_types = Counter()
    countries = set()
    all_ratings = []
    missing_ratings = 0
    
    for idx, movie in enumerate(data[:10]):  # Check first 10
        print(f"\nSample {idx}:")
        print(f"  Title: {movie.get('title', 'N/A')}")
        print(f"  Country: {movie.get('country', 'N/A')}")
        print(f"  Ratings: {movie.get('ratings', 'N/A')}")
        
    for movie in data:
        country = movie.get('country', 'Unknown')
        countries.add(country)
        
        ratings = movie.get('ratings', {})
        
        if isinstance(ratings, dict):
            rating_types['dict'] += 1
            rating = ratings.get('rating', 'Unknown')
            all_ratings.append(rating)
        elif isinstance(ratings, str):
            rating_types['string'] += 1
            all_ratings.append(ratings)
        else:
            rating_types['other'] += 1
            missing_ratings += 1
            all_ratings.append('Unknown')
    
    print(f"\nRating Format Distribution:")
    for fmt, count in rating_types.items():
        print(f"  {fmt}: {count} ({count/len(data)*100:.1f}%)")
    
    print(f"\nMissing/Invalid: {missing_ratings}")
    
    # Rating distribution
    print("\n" + "=" * 80)
    print("RATING DISTRIBUTION (Top 30)")
    print("=" * 80)
    
    rating_counts = Counter(all_ratings)
    for rating, count in rating_counts.most_common(30):
        pct = count / len(data) * 100
        print(f"  {rating:20s}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\nTotal Unique Ratings: {len(rating_counts)}")
    
    # Country distribution
    print("\n" + "=" * 80)
    print("COUNTRY DISTRIBUTION (Top 20)")
    print("=" * 80)
    
    country_counts = Counter([m.get('country', 'Unknown') for m in data])
    for country, count in country_counts.most_common(20):
        pct = count / len(data) * 100
        print(f"  {country:20s}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\nTotal Countries: {len(countries)}")
    
    # Class imbalance analysis
    print("\n" + "=" * 80)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 80)
    
    max_count = rating_counts.most_common(1)[0][1]
    min_count = rating_counts.most_common()[-1][1]
    
    print(f"Most common rating: {rating_counts.most_common(1)[0][0]} ({max_count} samples)")
    print(f"Least common rating: {rating_counts.most_common()[-1][0]} ({min_count} samples)")
    print(f"Imbalance ratio: {max_count/min_count:.1f}:1")
    
    # Check for long-tail
    total = len(data)
    cumsum = 0
    for idx, (rating, count) in enumerate(rating_counts.most_common(), 1):
        cumsum += count
        if cumsum / total >= 0.80:
            print(f"\nTop {idx} ratings cover 80% of data ({cumsum}/{total})")
            break
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if len(rating_counts) > 200:
        print("⚠️  WARNING: 200+ unique ratings detected")
        print("   → Consider grouping rare ratings or focusing on top N countries")
    
    if max_count / min_count > 100:
        print("⚠️  WARNING: Severe class imbalance (>100:1)")
        print("   → Use focal loss with high gamma (2.0-3.0)")
        print("   → Consider class-balanced sampling")
    
    if missing_ratings > len(data) * 0.01:
        print("⚠️  WARNING: >1% missing ratings")
        print("   → Filter out samples with missing ratings")
    
    print("\n✓ Diagnostic complete")

if __name__ == "__main__":
    # Find data file
    possible_paths = [
        Path("multimodal_expanded_coverage.json"),
        Path("/content/drive/MyDrive/multimodal_expanded_coverage.json"),
        Path("/content/drive/MyDrive/veridex_data/multimodal_expanded_coverage.json"),
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if not data_path:
        print("❌ ERROR: Cannot find multimodal_expanded_coverage.json")
        print("   Searched:")
        for path in possible_paths:
            print(f"   - {path}")
    else:
        diagnose_data(data_path)

