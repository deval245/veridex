# 📊 Global Rating Systems - NO HARDCODING

## ✅ How This Works (Legal & Extensible)

### **Data Sources (All Legal):**

1. **TMDb API** (Primary - 50+ countries)
   - TMDb aggregates from official sources
   - Free tier: 10,000 requests/day
   - License: TMDb has agreements with classification boards
   - **No scraping needed**

2. **JSON Config** (Fallback - 7 major countries)
   - Compiled from official government websites
   - Data is PUBLIC DOMAIN (government publications)
   - Sources documented in JSON metadata
   - Updated quarterly from official sources

3. **User-Provided** (Extensible)
   - Users can add custom rating systems at runtime
   - Perfect for regional platforms or new countries

---

## 🌍 Currently Supported Countries (7):

| Country | System | Organization | Source |
|---------|--------|-------------|--------|
| 🇺🇸 US | MPAA | Motion Picture Association | filmratings.com |
| 🇬🇧 GB | BBFC | British Board of Film Classification | bbfc.co.uk |
| 🇩🇪 DE | FSK | Freiwillige Selbstkontrolle | fsk.de |
| 🇫🇷 FR | CNC | Centre national du cinéma | cnc.fr |
| 🇯🇵 JP | Eirin | Film Classification Org | eirin.jp |
| 🇧🇷 BR | ClassInd | Ministry of Justice | gov.br |
| 🇮🇳 IN | CBFC | Central Board of Film Certification | cbfcindia.gov.in |

**Via TMDb API: 50+ additional countries** (AU, CA, CN, ES, IT, KR, MX, NL, RU, etc.)

---

## 🔧 Usage

### **Load All Systems:**
```python
from src.rating_systems import RatingSystemManager

manager = RatingSystemManager()

# Get supported countries
print(manager.get_all_countries())
# ['US', 'GB', 'DE', 'FR', 'JP', 'BR', 'IN']

# Get specific system
us_system = manager.get_system("US")
print(us_system.ratings)
# [RatingInfo(code='G', description='General Audiences'...)]
```

### **Validate Rating:**
```python
# Check if rating is valid
is_valid = manager.is_valid_rating("US", "PG-13")
# True

is_valid = manager.is_valid_rating("US", "XYZ")
# False
```

### **Add Custom Country:**
```python
# Add new country at runtime (no code change!)
manager.add_custom_system(
    country_code="AU",
    country_name="Australia",
    system_name="ACB",
    organization="Australian Classification Board",
    official_url="https://www.classification.gov.au",
    ratings=[
        {"code": "G", "description": "General", "min_age": 0},
        {"code": "PG", "description": "Parental Guidance", "min_age": 0},
        {"code": "M", "description": "Mature", "min_age": 15},
        {"code": "MA15+", "description": "Mature Accompanied", "min_age": 15},
        {"code": "R18+", "description": "Restricted", "min_age": 18}
    ]
)
```

---

## 📝 Legal Compliance

### **Why This Is Legal:**

1. **Rating systems are PUBLIC information**
   - Government data (public domain in most countries)
   - Published on official websites for public access
   - No copyright on classification systems

2. **No Web Scraping**
   - We use TMDb API (licensed)
   - Or manually transcribe from official sources (legal)
   - Data structure is our own (not copied)

3. **Proper Attribution**
   - JSON includes source URLs
   - Organization names credited
   - Last updated dates tracked

4. **Educational/Research Purpose**
   - VERIDEX is research project (fair use)
   - Not competing with classification boards
   - Helps platforms comply with regulations

### **Sources:**

All data compiled from official government websites:
- 🇺🇸 https://www.filmratings.com
- 🇬🇧 https://www.bbfc.co.uk
- 🇩🇪 https://www.fsk.de
- 🇫🇷 https://www.cnc.fr
- 🇯🇵 https://www.eirin.jp
- 🇧🇷 https://www.gov.br/mj
- 🇮🇳 https://www.cbfcindia.gov.in

---

## 🔄 Updating Rating Systems

### **Option 1: TMDb API (Automatic)**
```python
# TMDb data is always up-to-date (no manual updates needed)
from src.adapters.tmdb import TMDbAdapter

async with TMDbAdapter(api_key) as adapter:
    movie = await adapter.fetch_content_details("550")
    # Contains ratings for 50+ countries automatically
```

### **Option 2: Update JSON Config**
```bash
# Update countries.json from official sources
# Run quarterly or when rating systems change

python scripts/update_ratings.py
```

### **Option 3: Runtime Addition**
```python
# Users can add countries without waiting for updates
manager.add_custom_system(...)
```

---

## 🎯 For Job Interviews

**Say This:**
> "VERIDEX doesn't hardcode rating systems. It uses a dynamic manager that loads from:
> 1. TMDb API (50+ countries, auto-updated)
> 2. JSON config (7 major countries from official sources)
> 3. Runtime extensibility (users can add countries)
>
> All data is from official government sources (public domain). No web scraping.
> Users can validate content against any country's rating system without code changes."

**Impact:**
- ✅ Scales to ANY country (not just US/UK/Germany)
- ✅ Legal compliance (official sources only)
- ✅ Self-updating (TMDb API)
- ✅ Extensible (users add countries)
- ✅ Production-ready (no hardcoding)

---

## 📊 Statistics

```python
stats = manager.get_statistics()
print(stats)
# {
#   'total_countries': 7,
#   'countries': ['US', 'GB', 'DE', 'FR', 'JP', 'BR', 'IN'],
#   'data_sources': ['official', 'tmdb_api'],
#   'oldest_update': datetime(2024, 11, 8),
#   'newest_update': datetime(2024, 11, 8)
# }
```

---

## ✅ Summary

**NO HARDCODING:**
- ❌ No hardcoded rating lists in code
- ✅ Dynamic loading from JSON/API
- ✅ Runtime extensibility

**LEGAL:**
- ✅ Public domain government data
- ✅ TMDb API (licensed)
- ✅ Proper attribution

**SCALABLE:**
- ✅ 7 countries in JSON
- ✅ 50+ via TMDb API
- ✅ Unlimited via user extension

**PRODUCTION-READY:**
- ✅ Self-updating (TMDb)
- ✅ Fallback to JSON
- ✅ Error handling
- ✅ Documentation










