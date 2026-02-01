# Quick Start Guide

## 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo "OPENAI_API_KEY=your_key_here" > .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
echo "GEMINI_API_KEY=your_key_here" >> .env
```

Note: Only set keys for providers you intend to use.

## 2. Run Bias Experiment

The main experiment tests demographic bias using CV variants with identical qualifications but different names signaling race/gender.

```bash
# Run with default settings (10 iterations, GPT-4-turbo)
python scripts/run_triplet_experiment.py --provider openai --model gpt-4-turbo

# Custom iterations
python scripts/run_triplet_experiment.py --provider openai --model gpt-4-turbo --iterations 5

# Test with different providers
python scripts/run_triplet_experiment.py --provider anthropic --model claude-sonnet-4
python scripts/run_triplet_experiment.py --provider gemini --model gemini-2.0-flash
```

## 3. Analyze Results

After running experiments, generate visualizations and statistics:

```bash
python analyze_bias.py
```

Options:
```bash
python analyze_bias.py --no-plots          # Text analysis only
python analyze_bias.py --output ./my_figs  # Custom output directory
python analyze_bias.py --methodology       # Print methodology details
```

## 4. View Results

Results are saved in `results/<model_name>/`:
- `all_results.json` - Raw ratings data
- `analysis_summary.json` - Computed bias metrics

Figures are saved to `figures/`:
- `bias_heatmaps.png` - Bias comparisons across models/pipelines
- `quality_consistency.png` - Quality scores and rating consistency
- `criteria_bias_heatmap.png` - Criteria-level bias analysis
- `intersectionality_combined.png` - Ratings by demographic group
- `anonymized_quality_heatmap.png` - Anonymized vs identified CV quality
- `bias_vs_quality_scatter.png` - Bias-accuracy trade-off

## 5. Programmatic Usage

See `scripts/example_usage.py` for examples of using the framework programmatically:

```bash
python scripts/example_usage.py
```

## 6. Next Steps

- Modify prompts in `src/pipelines/*.py` to customize analysis
- Add new LLM providers in `src/providers/`
- Create new pipeline strategies in `src/pipelines/`
- Adjust settings in `config.yaml`
- Edit `data/cv_variants.json` to customize demographic variants
