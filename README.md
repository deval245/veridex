# VERIDEX 🔍

> **Universal API Compliance Validation powered by Multi-Agent AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**VERIDEX** is a production-grade multi-agent system for automated API compliance validation using LLM-powered reasoning.

## 🎯 What Problem Does VERIDEX Solve?

Traditional API validation is:
- ❌ **Manual**: Weeks of testing for large catalogs
- ❌ **Domain-specific**: Custom code for each use case
- ❌ **Brittle**: Hard-coded rules break with policy changes
- ❌ **Opaque**: No explanation when validation fails

VERIDEX makes validation:
- ✅ **Automated**: Minutes instead of weeks
- ✅ **Universal**: Works across any API domain
- ✅ **Adaptive**: Learns new rules automatically
- ✅ **Explainable**: Natural language reasoning for failures

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VERIDEX SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Discovery   │  │   Planning   │  │  Validation  │     │
│  │    Agent     │→ │    Agent     │→ │    Agent     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                  ↓                  ↓             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Knowledge Base (Neo4j + Weaviate)            │  │
│  │  • Policy documents  • Rating systems  • API specs   │  │
│  └──────────────────────────────────────────────────────┘  │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    Neuro-Symbolic Reasoning (LLM + Z3 Solver)        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/veridex.git
cd veridex
./setup.sh

# Add API keys to .env
# - OPENAI_API_KEY (https://platform.openai.com/api-keys)
# - TMDB_API_KEY (https://www.themoviedb.org/settings/api)

# Run demo
source venv/bin/activate
python examples/content_rating_demo.py
```

### Use in Code

```python
from src.veridex import VERIDEX
import asyncio

async def main():
    veridex = VERIDEX(tmdb_api_key="your_key")
    result = await veridex.validate_content_rating(
        movie_id="550", expected_rating="R", country="US"
    )
    print(f"{result.status}: {result.reason}")
    await veridex.cleanup()

asyncio.run(main())
```

## 📊 Features

- ✅ **Multi-agent orchestration** with timeout and retry logic
- ✅ **LLM-powered reasoning** for intelligent validation
- ✅ **Pluggable data sources** (TMDb, OMDb, custom APIs)
- ✅ **Async-first** for production performance
- ✅ **Type-safe** configuration via Pydantic
- ✅ **Zero hardcoding** - all behavior config-driven

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/test_agents.py

# Run with coverage
pytest --cov=src tests/
```

## 📁 Project Structure

```
veridex/
├── src/
│   ├── agents/          # Multi-agent system components
│   ├── core/            # Core validation logic
│   ├── knowledge/       # Knowledge base management
│   ├── validation/      # Domain-specific validators
│   └── utils/           # Utility functions
├── tests/               # Test suites
├── data/                # Datasets and benchmarks
├── examples/            # Usage examples
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks
└── configs/             # Configuration files
```

## 🔬 Technical Details

See `docs/ARCHITECTURE.md` for system design and implementation details.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

**Deval Thakkar**  
- Email: devalth8@gmail.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Built With

- [OpenAI GPT-4](https://openai.com) - Language understanding
- [Pydantic](https://pydantic.dev) - Data validation
- [aiohttp](https://docs.aiohttp.org) - Async HTTP client

---

**⭐ Star this repo if you find it useful!**

