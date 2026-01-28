# LLM CV Analysis Pipeline

A modular framework for comparing different LLM models and analysis strategies when evaluating CVs.

## Features

- **Multiple Pipeline Strategies**:
  - **One-Shot**: Single prompt with job ad and all CVs, direct ranking output
  - **Chain-of-Thought**: Step-by-step reasoning through each criteria, then final ranking
  - **Multi-Layer**: Evaluate each criteria separately via LLM, then LLM synthesizes overall fit
  - **Decomposed Algorithmic**: Evaluate each criteria separately via LLM, then algorithmically aggregate (simple average) - designed to reduce bias

- **Job Ad Integration**: Pipelines evaluate CVs against a specific job description and detailed hiring criteria

- **Ranking Output**: Each pipeline outputs rankings (1-4) for all candidates with names and reasoning

- **Multiple LLM Providers**: Support for OpenAI (GPT-4, GPT-5), Google Gemini, and Anthropic (Claude) models
- **Easy LLM Switching**: Abstract provider interface makes it simple to switch between LLM providers

- **Comparison Framework**: Built-in tools to compare results across pipelines and models

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
Note: You only need to set the API keys for providers you want to use.

3. **Sanitize CV data** (first time only):
```bash
python sanitize_cvs.py
```
This creates `data/cvs_sanitized.json` with randomized IDs and a mapping file.

4. **Configure settings** (optional):
Edit `config.yaml` to customize models, pipelines, and analysis settings.

## Usage

### Basic Usage

Run all pipelines on all CVs with all configured models:
```bash
python run_analysis.py
```

**Note**: Each pipeline evaluates ALL CVs at once and outputs rankings (1-4) for each candidate.

### Quick Test

Run a quick test on C and D CVs (C1, C2, C3, D1, D2):
```bash
python run_analysis.py --quick-test
```

### Extended Test

Run an extended test on A, B, C, and D CVs (A1-A3, B1-B2, C1-C3, D1-D2):
```bash
python run_analysis.py --extended-test --experiment-name extended_test
```

### Custom Experiments

Run specific models:
```bash
python run_analysis.py --models gpt-4o-mini gpt-4o
```

Run specific providers:
```bash
python run_analysis.py --providers gemini anthropic
```

Combine providers and models:
```bash
python run_analysis.py --providers openai gemini --models gpt-4o gemini-1.5-pro
```

Run specific pipelines:
```bash
python run_analysis.py --pipelines one_shot chain_of_thought
```

Compare multi_layer with decomposed_algorithmic (bias comparison):
```bash
python run_analysis.py --pipelines multi_layer decomposed_algorithmic --quick-test
```

Analyze specific CVs:
```bash
python run_analysis.py --cv-ids A1 A2 B1
```

Combine options:
```bash
python run_analysis.py --models gpt-4o-mini --pipelines one_shot --cv-ids A1 A2 --experiment-name my_test
```

## Project Structure

```
.
├── src/
│   ├── providers/          # LLM provider implementations
│   │   ├── base.py         # Abstract base class
│   │   └── openai_provider.py
│   ├── pipelines/          # Analysis pipeline strategies
│   │   ├── base.py
│   │   ├── one_shot.py
│   │   ├── chain_of_thought.py
│   │   ├── multi_layer.py
│   │   └── decomposed_algorithmic.py
│   └── comparison.py       # Comparison and evaluation framework
├── data/                   # CV data files
├── results/                # Experiment results (generated)
├── config.yaml            # Configuration file
├── requirements.txt        # Python dependencies
└── run_analysis.py        # Main execution script
```

## Adding New LLM Providers

1. Create a new provider class in `src/providers/` that inherits from `LLMProvider`
2. Implement the `generate()` and `get_provider_name()` methods
3. Add it to `src/providers/__init__.py`
4. Update `run_analysis.py` to support the new provider

Example:
```python
from src.providers.base import LLMProvider, LLMResponse

class MyNewProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Your implementation
        pass
    
    def get_provider_name(self) -> str:
        return "my_provider"
```

## Adding New Pipeline Strategies

1. Create a new pipeline class in `src/pipelines/` that inherits from `Pipeline`
2. Implement the `analyze()` method
3. Add it to `src/pipelines/__init__.py`
4. Update `run_analysis.py` to support the new pipeline

## Results

Results are saved in the `results/` directory with:
- JSON files for each pipeline/model combination (contains all CV rankings)
- `*_rankings.txt` files with human-readable rankings (names and ratings 1-4)
- `summary.json` with experiment overview
- `comparison.csv` for easy analysis in Excel/Pandas

### Ranking System
- **4** = Excellent fit (meets all criteria at excellent level)
- **3** = Good fit (meets criteria at good level)
- **2** = Borderline fit (meets some criteria but has gaps)
- **1** = Not a fit (does not meet key criteria)

## Analyzing Differences

After running an experiment, analyze how each CV is treated differently across models and pipelines:

```bash
# Analyze most recent experiment
python analyze_differences.py

# Analyze specific experiment
python analyze_differences.py quick_test_run
```

This generates:
- Detailed analysis showing variance in rankings for each CV
- Breakdown by pipeline and model
- Summary table sorted by disagreement (variance)
- Pivot table showing rankings across all pipeline-model combinations
- CSV files for further analysis

## Configuration

Edit `config.yaml` to customize:
- Available LLM models
- Pipeline settings
- Temperature and token limits
- Results directory

