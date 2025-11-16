# TMDb Attribution for arXiv Paper

## Quick Reference for Paper Writing

### 1. Data Section (Methods/Dataset)

**Recommended Text:**

```
Dataset:
We construct a dataset of 12,264 movies with 40,610 rating samples across 
65 countries and 51 rating classes. Movie metadata (titles, synopses, release 
dates) was obtained from The Movie Database (TMDb) API [1]. Content ratings 
were collected from public sources for each country's rating system.

[1] The Movie Database (TMDb). https://www.themoviedb.org/
    This product uses the TMDb API but is not endorsed or certified by TMDb.
```

### 2. Acknowledgments Section

**Required Text:**

```
Acknowledgments:
We thank The Movie Database (TMDb) for providing movie metadata through 
their API. This product uses the TMDb API but is not endorsed or certified 
by TMDb.
```

### 3. BibTeX Citation (Optional but Recommended)

```bibtex
@misc{tmdb2024,
  title={The Movie Database (TMDb)},
  author={{TMDb}},
  year={2024},
  url={https://www.themoviedb.org/},
  note={This product uses the TMDb API but is not endorsed or certified by TMDb}
}
```

### 4. Compliance Statement (if needed in paper)

**For Methods/Data Section:**

```
Data Collection and Compliance:
All movie metadata was collected through the official TMDb API using a 
registered API key. Data collection complied with TMDb's terms of service, 
including rate limits (40 requests/second, 100,000 requests/day). Only 
public domain metadata (titles, synopses, release dates) was used; no 
copyrighted images or content were included. The dataset is used exclusively 
for academic research and ML model training, in compliance with TMDb's 
research use terms.
```

---

## ✅ Checklist for Paper Submission

- [ ] TMDb mentioned in Data/Dataset section
- [ ] TMDb attribution text included (required phrase)
- [ ] TMDb URL included
- [ ] Acknowledgments section includes TMDb
- [ ] Compliance statement (if required by journal)
- [ ] BibTeX citation (optional but professional)

---

## 📝 Key Points to Emphasize

1. **Public Domain Only**: Only factual metadata (titles, synopses), not copyrighted content
2. **Official API**: Used official API, not scraping
3. **Research Purpose**: Academic research, not commercial redistribution
4. **Transformative Use**: Data used to train ML model, not republished as-is
5. **Compliance**: Followed all TMDb terms and rate limits

---

## 🚫 What NOT to Say

- ❌ "We scraped TMDb" (we used official API)
- ❌ "We downloaded TMDb images" (we only have URLs, not images)
- ❌ "We redistributed TMDb data" (we only use it for training)
- ❌ "TMDb endorsed this research" (explicitly not endorsed)

---

## ✅ Safe Statements

- ✅ "We obtained movie metadata from TMDb API"
- ✅ "We used TMDb data for research purposes"
- ✅ "Dataset contains TMDb metadata (titles, synopses)"
- ✅ "Compliant with TMDb terms of service"
- ✅ "Transformative use for ML training"

---

**Last Updated**: 2024-11-XX
