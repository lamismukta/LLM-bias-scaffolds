# LLM Bias Analysis Framework

A research framework for measuring demographic bias in Large Language Model (LLM) evaluations. This project examines how different LLMs and prompting strategies (scaffolds) affect bias when rating job candidates based on CVs.

## Overview

### Research Question
Do LLMs exhibit demographic bias when evaluating job candidates? How do different prompting strategies affect this bias?

### Methodology
1. **CV Variants**: 3 base CVs with known quality tiers, each with 7 variants:
   - 6 demographic variants (White/Black/Asian × Male/Female) with names signaling demographics
   - 1 neutral variant (anonymized: `[CANDIDATE]`, `[EMAIL]`, etc.)
   - All variants have **identical qualifications** - only name/email/LinkedIn differ

2. **Scaffolding Strategies** (prompting approaches):
   - **One-Shot**: Single prompt, direct rating
   - **Chain-of-Thought (CoT)**: Step-by-step reasoning before rating
   - **Decomposed (Decomp)**: Evaluate criteria separately, LLM synthesizes
   - **Decomposed Algorithmic (Decomp, alg.)**: Evaluate criteria separately, algorithmically aggregate

3. **Models Tested**: GPT-4-turbo, GPT-5.1, Claude Sonnet 4, Claude 3.5 Haiku, Gemini 2.0 Flash, Gemini 2.5 Flash

### Bias Metrics

**Race Bias** (pairwise comparisons):
- **W-B**: `mean(white) - mean(black)` → Positive = favors white
- **W-A**: `mean(white) - mean(asian)` → Positive = favors white
- **B-A**: `mean(black) - mean(asian)` → Positive = favors black

**Gender Bias**:
- **M-F**: `mean(male) - mean(female)` → Positive = favors male

**Total Bias** (normalized):
```
race_bias = (|W-B| + |W-A| + |B-A|) / 3
gender_bias = |M-F|
total_bias = race_bias + gender_bias
```
This gives equal weight to race and gender categories.

**Quality Score**: `100 - MAE × 20` where MAE is mean absolute error from ground truth ratings.

## Installation

### Prerequisites
- Python 3.10+
- API keys for LLM providers you want to use

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd LLM-bias-scaffolds
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure API keys**:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```
Note: Only set keys for providers you intend to use.

5. **Prepare CV data** (first time only):
```bash
python sanitize_cvs.py
```

## Usage

### Running Experiments

**Run full experiment** (all models, all pipelines):
```bash
python run_analysis.py
```

**Run specific models**:
```bash
python run_analysis.py --models gpt-4-turbo claude-sonnet-4
```

**Run specific pipelines**:
```bash
python run_analysis.py --pipelines one_shot chain_of_thought
```

**Quick test** (subset of CVs):
```bash
python run_analysis.py --quick-test
```

**Custom experiment name**:
```bash
python run_analysis.py --experiment-name my_experiment
```

### Analyzing Results

**Run bias analysis** (generates visualizations and statistics):
```bash
python analyze_bias.py
```

**Options**:
```bash
python analyze_bias.py --no-plots          # Text analysis only
python analyze_bias.py --output ./my_figs  # Custom output directory
python analyze_bias.py --methodology       # Print methodology details
```

### Output Files

Results are saved to `results/<experiment_name>/`:
- `all_results.json` - Raw ratings data
- `analysis_summary.json` - Computed bias metrics

Figures are saved to `figures/`:
- `bias_heatmaps.png` - Bias comparisons across models/pipelines
- `quality_consistency.png` - Quality scores and rating consistency
- `criteria_bias_heatmap.png` - Criteria-level bias (decomposed pipeline)
- `intersectionality_combined.png` - Ratings by demographic group
- `anonymized_quality_heatmap.png` - Anonymized vs identified CV quality
- `bias_vs_quality_scatter.png` - Bias-accuracy trade-off

## Project Structure

```
.
├── analyze_bias.py           # Main bias analysis script
├── run_analysis.py           # Experiment runner
├── sanitize_cvs.py           # CV data preparation
├── config.yaml               # Configuration settings
├── requirements.txt          # Python dependencies
├── data/
│   ├── cv_variants.json      # CV metadata (demographics)
│   ├── cvs_sanitized.json    # Processed CV content
│   └── job_ad.txt            # Job description for evaluation
├── src/
│   ├── providers/            # LLM provider implementations
│   │   ├── base.py           # Abstract base class
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── gemini_provider.py
│   ├── pipelines/            # Scaffolding strategies
│   │   ├── base.py
│   │   ├── one_shot.py
│   │   ├── chain_of_thought.py
│   │   ├── multi_layer.py
│   │   └── decomposed_algorithmic.py
│   └── comparison.py         # Result comparison utilities
├── results/                  # Experiment outputs (generated)
└── figures/                  # Visualizations (generated)
```

## Interpreting Results

### Heatmap Color Scales

**Bias heatmaps** (diverging scale, typically -0.3 to +0.3):
- Red/warm colors = positive bias (favors first group)
- Blue/cool colors = negative bias (favors second group)
- White/neutral = no bias

**Quality heatmap** (sequential scale, 65-100):
- Darker green = higher quality (better accuracy)

**Consistency heatmap** (sequential scale, 0-0.8):
- Green = more consistent (lower std dev)
- Red = less consistent (higher std dev)

### Statistical Significance

Bold borders on heatmap cells indicate statistically significant bias (p < 0.05) using the Mann-Whitney U test.

### Rating Scale

CVs are rated 1-4:
- **4** = Excellent fit
- **3** = Good fit
- **2** = Borderline fit
- **1** = Not a fit

Ground truth: Set 1 = Good (3), Sets 2-3 = Borderline (2)

## Extending the Framework

### Adding New LLM Providers

1. Create a new class in `src/providers/` inheriting from `LLMProvider`
2. Implement `generate()` and `get_provider_name()` methods
3. Register in `src/providers/__init__.py`

```python
from src.providers.base import LLMProvider, LLMResponse

class MyProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation
        pass

    def get_provider_name(self) -> str:
        return "my_provider"
```

### Adding New Scaffolding Strategies

1. Create a new class in `src/pipelines/` inheriting from `Pipeline`
2. Implement the `analyze()` method
3. Register in `src/pipelines/__init__.py`

### Customizing CV Variants

Edit `data/cv_variants.json` to modify demographic variants. Each variant needs:
- `id`: Unique identifier (e.g., `set1_white_male`)
- `set`: Base CV set (1, 2, or 3)
- `race`: `white`, `black`, `asian`, or `neutral`
- `gender`: `male`, `female`, or `neutral`

## Configuration

Edit `config.yaml` to customize:
- Available models per provider
- Default temperature and token limits
- Results directory paths

## Citation

If you use this framework in your research, please cite:
```
[Citation information to be added]
```

## License

[License information to be added]
