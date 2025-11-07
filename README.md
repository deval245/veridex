# VERIDEX 🔍

> **Universal API Compliance Validation powered by Multi-Agent AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](https://arxiv.org)

**VERIDEX** is a self-evolving multi-agent system that autonomously validates API compliance across any domain using neuro-symbolic reasoning.

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

### Prerequisites

- Python 3.9+
- OpenAI API key (or Claude/Llama)
- 8GB RAM minimum

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/veridex.git
cd veridex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run Demo

```bash
# Run content rating validation demo
python examples/content_rating_demo.py

# Run with custom data
python examples/validate_api.py --input data/my_content.csv
```

## 📊 Example Use Cases

### 1. Content Rating Validation (Demo)

```python
from veridex import VERIDEX
from veridex.adapters import TMDbAdapter

# Initialize VERIDEX
validator = VERIDEX(
    data_source=TMDbAdapter(api_key="your_tmdb_key"),
    domain="content_ratings"
)

# Validate content
results = validator.validate(
    content_ids=["550", "deadpool"],
    regions=["US", "BR", "UK"]
)

# Get detailed report
results.to_csv("validation_report.csv")
print(results.summary())
```

### 2. Financial Compliance (Coming Soon)

```python
validator = VERIDEX(
    data_source=YourBankAPI(),
    domain="aml_compliance"
)
```

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

## 🎓 Research

**Paper**: Coming soon on arXiv  
**Benchmark**: PolicyBench dataset (in preparation)

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

**Deval Thakkar**  
- Email: devalth8@gmail.com
- GitHub: [@yourusername](https://github.com/yourusername)
- Website: [veridex.cloud](https://veridex.cloud)

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) for agent orchestration
- [OpenAI GPT-4](https://openai.com) for language understanding
- [Neo4j](https://neo4j.com) for knowledge graphs
- [Z3 Solver](https://github.com/Z3Prover/z3) for formal verification

---

**⭐ Star this repo if you find it useful!**

