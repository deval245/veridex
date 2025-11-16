# Data Acquisition Guide - VERIDEX V9.1

## ⚠️ Dataset Not Included

This repository does **NOT** include the training dataset (`multimodal_expanded_coverage.json`) due to:

1. **File Size**: 9.8 MB (approaching GitHub limits)
2. **TMDb Licensing**: Contains TMDb-derived metadata subject to API terms
3. **Publication Standards**: Top-tier research repositories exclude datasets
4. **Reproducibility**: Dataset can be recreated using provided scripts

---

## 📋 Dataset Specifications

**Multimodal Expanded Coverage Dataset**
- **Movies**: 12,264
- **Samples**: 40,610 (after filtering)
- **Countries**: 65
- **Rating Classes**: 51
- **Rating Systems**: MPAA, BBFC, FSK, CBFC, Eirin, ACB, CNC, DJCTQ, etc.
- **Split**: 80% train / 10% val / 10% test (seed=42)

---

## 🔄 How to Obtain the Dataset

### Option 1: Recreate from TMDb API (Recommended)

1. **Get TMDb API Key**
   - Register at https://www.themoviedb.org/
   - Obtain API key from https://www.themoviedb.org/settings/api
   - Store in environment variable: `export TMDB_API_KEY=your_key`

2. **Run Data Collection Script**
   ```bash
   # Use the data collection scripts (if available)
   # Or follow TMDb API documentation to collect:
   # - Movie titles, synopses, release dates
   # - Content ratings from public sources
   ```

3. **Compliance Requirements**
   - Follow TMDb API rate limits: 40 requests/second, 100,000/day
   - Include attribution: "This product uses the TMDb API but is not endorsed or certified by TMDb"
   - See [TMDB_COMPLIANCE.md](TMDB_COMPLIANCE.md) for details

### Option 2: Request from Author

For research purposes, you may contact the author:
- **Email**: deval.thakkar.research@protonmail.com
- **Subject**: "VERIDEX V9.1 Dataset Request - Research Use"

**Note**: Dataset sharing is subject to:
- Academic/research use only
- TMDb API terms compliance
- No redistribution without permission

---

## 📊 Dataset Structure

The dataset follows this structure:

```json
{
  "metadata": {
    "created_at": "20251111_120529",
    "total_movies": 12264,
    "version": "expanded-coverage",
    "countries": [...],
    "rating_systems": [...]
  },
  "movies": [
    {
      "tmdb_id": 12345,
      "title": "Movie Title",
      "synopsis": "Movie description...",
      "release_date": "2020-01-01",
      "ratings": {
        "US": {"MPAA": "R", "label": "R"},
        "GB": {"BBFC": "15", "label": "15"},
        ...
      }
    },
    ...
  ]
}
```

---

## ✅ Verification

After obtaining the dataset:

1. **Place in correct location**:
   ```bash
   data/multimodal_expanded_coverage.json
   ```

2. **Verify structure**:
   ```python
   import json
   with open('data/multimodal_expanded_coverage.json') as f:
       data = json.load(f)
   assert 'metadata' in data
   assert 'movies' in data
   assert data['metadata']['total_movies'] == 12264
   print("✓ Dataset structure valid")
   ```

3. **Run training script**:
   ```bash
   python TRAIN_V9.1_ULTIMATE.py
   ```

---

## 📄 TMDb Attribution

When using this dataset, you must:

1. **Attribute TMDb**: Include in your paper/repository:
   > "This product uses the TMDb API but is not endorsed or certified by TMDb."

2. **Comply with Terms**: See [TMDB_COMPLIANCE.md](TMDB_COMPLIANCE.md)

3. **Rate Limits**: Respect TMDb API rate limits

---

## 🔒 License Restrictions

The dataset is subject to:
- **VERIDEX Research License**: See [LICENSE](LICENSE)
- **TMDb API Terms**: https://www.themoviedb.org/documentation/api/terms-of-use
- **No Commercial Use**: Dataset cannot be used for commercial purposes
- **No Redistribution**: Cannot redistribute without written permission

---

**Last Updated**: November 2024

