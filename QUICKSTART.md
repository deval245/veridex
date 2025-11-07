# VERIDEX Quick Start

## Setup (2 minutes)

```bash
# 1. Run setup script
./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Add API keys to .env
# Get TMDb key: https://www.themoviedb.org/settings/api
# Get OpenAI key: https://platform.openai.com/api-keys
```

## Run Demo

```bash
python examples/content_rating_demo.py
```

## Use in Code

```python
from src.veridex import VERIDEX
import asyncio

async def validate():
    veridex = VERIDEX(tmdb_api_key="your_key")
    
    result = await veridex.validate_content_rating(
        movie_id="550",
        expected_rating="R",
        country="US"
    )
    
    print(f"Status: {result.status}")
    print(f"Reason: {result.reason}")
    await veridex.cleanup()

asyncio.run(validate())
```

## Project Structure

```
src/
├── veridex.py          # Main API
├── config.py           # Settings
├── agents/             # Agent implementations
├── core/               # Core logic (LLM, orchestrator, types)
└── adapters/           # Data source adapters

examples/               # Usage examples
tests/                  # Test suites
```

## Next Steps

1. Run demo to verify setup
2. Check `examples/` for more use cases
3. Read `docs/ARCHITECTURE.md` for details
4. Start building your validation logic

