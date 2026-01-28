#!/usr/bin/env python3
"""Main script to run CV analysis pipelines and compare results."""
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.providers.openai_provider import OpenAIProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.anthropic_provider import AnthropicProvider
from src.pipelines.one_shot import OneShotPipeline
from src.pipelines.chain_of_thought import ChainOfThoughtPipeline
from src.pipelines.multi_layer import MultiLayerPipeline
from src.pipelines.decomposed_algorithmic import DecomposedAlgorithmicPipeline
from src.comparison import ComparisonFramework
from src.pipelines.base import PipelineResult
from src.job_data import load_job_ad, load_detailed_criteria


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_cv_data(data_path: str) -> List[Dict[str, Any]]:
    """Load CV data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_provider_for_model(model: str, config: Dict[str, Any]):
    """Determine which provider to use for a given model."""
    # Check each provider's model list
    for provider_name, provider_config in config['llm_providers'].items():
        if model in provider_config.get('models', []):
            if provider_name == 'openai':
                return OpenAIProvider(
                    model=model,
                    temperature=provider_config.get('temperature', 1.0),
                    max_tokens=provider_config.get('max_tokens', 2000)
                )
            elif provider_name == 'gemini':
                return GeminiProvider(
                    model=model,
                    temperature=provider_config.get('temperature', 1.0),
                    max_tokens=provider_config.get('max_tokens', 2000)
                )
            elif provider_name == 'anthropic':
                return AnthropicProvider(
                    model=model,
                    temperature=provider_config.get('temperature', 1.0),
                    max_tokens=provider_config.get('max_tokens', 2000)
                )

    # Fallback: detect provider by model name prefix
    model_lower = model.lower()
    if model_lower.startswith('gemini'):
        provider_config = config['llm_providers'].get('gemini', {})
        return GeminiProvider(
            model=model,
            temperature=provider_config.get('temperature', 1.0),
            max_tokens=provider_config.get('max_tokens', 2000)
        )
    elif model_lower.startswith('claude'):
        provider_config = config['llm_providers'].get('anthropic', {})
        return AnthropicProvider(
            model=model,
            temperature=provider_config.get('temperature', 1.0),
            max_tokens=provider_config.get('max_tokens', 2000)
        )

    # Default to OpenAI if not found
    return OpenAIProvider(
        model=model,
        temperature=config['llm_providers']['openai'].get('temperature', 1.0),
        max_tokens=config['llm_providers']['openai'].get('max_tokens', 2000)
    )


async def run_experiment(
    config: Dict[str, Any],
    cv_data: List[Dict[str, Any]],
    job_ad: str,
    detailed_criteria: str,
    models: List[str] = None,
    pipelines: List[str] = None,
    cv_ids: List[str] = None,
    providers: List[str] = None
) -> List[PipelineResult]:
    """Run a complete experiment across models and pipelines."""
    load_dotenv()
    
    # Filter CVs if specified
    if cv_ids:
        cv_data = [cv for cv in cv_data if cv['id'] in cv_ids]
    
    # Collect all models from all providers if not specified
    if models is None:
        models = []
        if providers is None:
            # Include all providers
            providers = list(config['llm_providers'].keys())
        
        for provider_name in providers:
            if provider_name in config['llm_providers']:
                models.extend(config['llm_providers'][provider_name].get('models', []))
    elif providers is not None:
        # Filter models by provider if both specified
        provider_models = []
        for provider_name in providers:
            if provider_name in config['llm_providers']:
                provider_models.extend(config['llm_providers'][provider_name].get('models', []))
        models = [m for m in models if m in provider_models]
    
    if pipelines is None:
        pipelines = [name for name, settings in config['pipelines'].items() if settings.get('enabled', True)]
    
    results = []
    total_tasks = len(models) * len(pipelines)
    completed = 0
    
    print(f"Running experiment:")
    print(f"  Models: {models}")
    print(f"  Pipelines: {pipelines}")
    print(f"  CVs: {len(cv_data)}")
    print(f"  Total pipeline runs: {total_tasks}\n")
    
    for model in models:
        # Create provider for this model (auto-detect based on model name)
        try:
            provider = get_provider_for_model(model, config)
        except Exception as e:
            print(f"  ✗ Error creating provider for {model}: {e}")
            continue
        
        for pipeline_name in pipelines:
            # Create pipeline
            if pipeline_name == "one_shot":
                pipeline = OneShotPipeline(provider)
            elif pipeline_name == "chain_of_thought":
                pipeline = ChainOfThoughtPipeline(provider)
            elif pipeline_name == "multi_layer":
                pipeline = MultiLayerPipeline(provider)
            elif pipeline_name == "decomposed_algorithmic":
                pipeline = DecomposedAlgorithmicPipeline(provider)
            else:
                print(f"Unknown pipeline: {pipeline_name}")
                continue
            
            # Run pipeline on all CVs at once
            print(f"Running {pipeline_name} with {model} on {len(cv_data)} CVs...")
            try:
                result = await pipeline.analyze(cv_data, job_ad, detailed_criteria)
                results.append(result)
                completed += 1
                print(f"  ✓ Completed ({completed}/{total_tasks})")
                print(f"    Rankings: {len(result.rankings)} CVs evaluated\n")
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"  ✗ Error: {e}\n")
                import traceback
                traceback.print_exc()
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}\n")
                import traceback
                traceback.print_exc()
    
    return results


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM CV analysis pipelines")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data", default="data/cvs_sanitized.json", help="Path to sanitized CV data file")
    parser.add_argument("--models", nargs="+", help="Specific models to test (default: all in config)")
    parser.add_argument("--providers", nargs="+", choices=["openai", "gemini", "anthropic"],
                       help="Specific providers to test (default: all)")
    parser.add_argument("--pipelines", nargs="+", choices=["one_shot", "chain_of_thought", "multi_layer", "decomposed_algorithmic"],
                       help="Specific pipelines to run (default: all enabled)")
    parser.add_argument("--cv-ids", nargs="+", help="Specific CV IDs to analyze (default: all)")
    parser.add_argument("--experiment-name", help="Name for this experiment (default: auto-generated with timestamp)")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test on C and D CVs (C1-C3, D1-D2)")
    parser.add_argument("--extended-test", action="store_true", help="Run extended test on A, B, C, and D CVs (A1-A3, B1-B2, C1-C3, D1-D2)")
    parser.add_argument("--small-test", action="store_true", help="Run small test on 4 matched CVs (A1, A2, A3, B1)")
    
    args = parser.parse_args()
    
    # Check if sanitized CVs exist
    if not Path(args.data).exists():
        print(f"Error: {args.data} not found.")
        print("Please run sanitize_cvs.py first to create sanitized CV data.")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load CV data
    cv_data = load_cv_data(args.data)
    
    # Small test mode - filter for 4 matched CVs (A1, A2, A3, B1)
    if args.small_test:
        # Load mapping to find sanitized IDs for A1, A2, A3, B1
        mapping_path = Path(__file__).parent / "data" / "cv_id_mapping.json"
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Find sanitized IDs for A1, A2, A3, B1 (4 matched CVs)
        target_original_ids = ['A1', 'A2', 'A3', 'B1']
        sanitized_ids = []
        for sanitized_id, info in mapping.items():
            if info['original_id'] in target_original_ids:
                sanitized_ids.append(sanitized_id)
        
        # Filter CVs
        cv_data = [cv for cv in cv_data if cv['id'] in sanitized_ids]
        print(f"Running in small test mode (4 matched CVs: {len(cv_data)} CVs)")
        print()
    
    # Quick test mode - filter for C and D CVs
    if args.quick_test:
        # Load mapping to find sanitized IDs for C1, C2, C3, D1, D2
        mapping_path = Path(__file__).parent / "data" / "cv_id_mapping.json"
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Find sanitized IDs for C1, C2, C3, D1, D2
        target_original_ids = ['C1', 'C2', 'C3', 'D1', 'D2']
        sanitized_ids = []
        for sanitized_id, info in mapping.items():
            if info['original_id'] in target_original_ids:
                sanitized_ids.append(sanitized_id)
        
        # Filter CVs
        cv_data = [cv for cv in cv_data if cv['id'] in sanitized_ids]
        print(f"Running in quick test mode (C and D CVs only: {len(cv_data)} CVs)")
        print()
    
    # Extended test mode - filter for A, B, C, and D CVs
    if args.extended_test:
        # Load mapping to find sanitized IDs for A1-A3, B1-B2, C1-C3, D1-D2
        mapping_path = Path(__file__).parent / "data" / "cv_id_mapping.json"
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Find sanitized IDs for A1, A2, A3, B1, B2, C1, C2, C3, D1, D2
        target_original_ids = ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'C3', 'D1', 'D2']
        sanitized_ids = []
        for sanitized_id, info in mapping.items():
            if info['original_id'] in target_original_ids:
                sanitized_ids.append(sanitized_id)
        
        # Filter CVs
        cv_data = [cv for cv in cv_data if cv['id'] in sanitized_ids]
        print(f"Running in extended test mode (A, B, C, and D CVs: {len(cv_data)} CVs)")
        print()
    
    # Load job ad and criteria
    job_ad = load_job_ad()
    detailed_criteria = load_detailed_criteria()
    
    # Run experiment
    results = await run_experiment(
        config=config,
        cv_data=cv_data,
        job_ad=job_ad,
        detailed_criteria=detailed_criteria,
        models=args.models,
        pipelines=args.pipelines,
        cv_ids=args.cv_ids,
        providers=args.providers
    )
    
    if not results:
        print("No results to save.")
        return
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        from datetime import datetime
        # Create descriptive name based on test type and timestamp
        test_type = ""
        if args.extended_test:
            test_type = "extended"
        elif args.quick_test:
            test_type = "quick"
        else:
            test_type = "full"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{test_type}_test_{timestamp}"
        print(f"\n{'='*60}")
        print(f"Auto-generated experiment name: {args.experiment_name}")
        print(f"{'='*60}\n")
    else:
        print(f"\nUsing specified experiment name: {args.experiment_name}\n")
    
    # Save and compare results
    framework = ComparisonFramework(results_dir=config.get('analysis', {}).get('results_dir', 'results'))
    experiment_dir = framework.save_results(results, experiment_name=args.experiment_name)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    comparison_df = framework.create_comparison_dataframe(results)
    
    print("\nRanking Distribution by Pipeline:")
    print(comparison_df.groupby(['pipeline', 'ranking_label']).size().unstack(fill_value=0))
    
    print("\nToken Usage by Pipeline:")
    usage_summary = comparison_df.groupby('pipeline')['total_tokens'].agg(['mean', 'sum', 'count'])
    print(usage_summary)
    
    print(f"\nDetailed results saved to: {experiment_dir}")
    print(f"Comparison CSV: {experiment_dir / 'comparison.csv'}")
    print(f"Rankings files: {experiment_dir / '*_rankings.txt'}")


if __name__ == "__main__":
    asyncio.run(main())
