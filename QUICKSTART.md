# Quick Start Guide

## 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

## 2. Run Your First Experiment

### Quick Test (5 CVs, all pipelines, default models)
```bash
python run_analysis.py --quick-test
```

### Full Experiment
```bash
python run_analysis.py
```

### Custom Experiment
```bash
# Test one pipeline on specific CVs
python run_analysis.py --pipelines one_shot --cv-ids A1 A2 --experiment-name test_one_shot

# Compare two models
python run_analysis.py --models gpt-4o-mini gpt-4o --pipelines one_shot chain_of_thought
```

## 3. View Results

Results are saved in `results/[experiment_name]/`:
- `comparison.csv` - Easy to open in Excel/Pandas
- `summary.json` - Experiment overview
- Individual JSON files for each CV/pipeline/model combination

## 4. Programmatic Usage

See `scripts/example_usage.py` for examples of using the framework programmatically:

```bash
python scripts/example_usage.py
```

## 5. Next Steps

- Modify prompts in `src/pipelines/*.py` to customize analysis
- Add new LLM providers in `src/providers/`
- Create new pipeline strategies in `src/pipelines/`
- Adjust settings in `config.yaml`

