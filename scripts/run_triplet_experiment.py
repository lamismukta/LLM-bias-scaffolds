#!/usr/bin/env python3
"""
Run bias experiment with CV variants.

Tests whether identical qualifications get different ratings based on demographic signals.

Structure:
- 3 CV sets (1 Good tier, 2 Borderline tier)
- 7 demographic variants each (3 races × 2 genders + neutral)
- 4 scaffolding pipelines
- N iterations per model

Total: 21 variants × 4 pipelines × N iterations × M models
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Ensure we're working from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)


# Scaffolds to test
PIPELINES = ['one_shot', 'chain_of_thought', 'multi_layer', 'decomposed_algorithmic']


def get_provider(config: dict, provider_name: str = 'openai', model: str = 'gpt-4-turbo'):
    """Create LLM provider."""
    if provider_name == 'openai':
        from src.providers.openai_provider import OpenAIProvider
        provider_config = config['llm_providers'].get('openai', {})
        temp = provider_config.get('temperature', 1.0)
        max_tokens = provider_config.get('max_tokens', 2000)
        return OpenAIProvider(model=model, temperature=temp, max_tokens=max_tokens)
    elif provider_name == 'anthropic':
        from src.providers.anthropic_provider import AnthropicProvider
        provider_config = config['llm_providers'].get('anthropic', {})
        temp = provider_config.get('temperature', 1.0)
        max_tokens = provider_config.get('max_tokens', 2000)
        return AnthropicProvider(model=model, temperature=temp, max_tokens=max_tokens)
    elif provider_name == 'gemini':
        from src.providers.gemini_provider import GeminiProvider
        provider_config = config['llm_providers'].get('gemini', {})
        temp = provider_config.get('temperature', 1.0)
        max_tokens = provider_config.get('max_tokens', 2000)
        return GeminiProvider(model=model, temperature=temp, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def get_pipeline(pipeline_name: str, provider):
    """Create pipeline instance."""
    from src.pipelines.one_shot import OneShotPipeline
    from src.pipelines.chain_of_thought import ChainOfThoughtPipeline
    from src.pipelines.multi_layer import MultiLayerPipeline
    from src.pipelines.decomposed_algorithmic import DecomposedAlgorithmicPipeline

    if pipeline_name == 'one_shot':
        return OneShotPipeline(provider, blind_mode=False)
    elif pipeline_name == 'chain_of_thought':
        return ChainOfThoughtPipeline(provider, blind_mode=False)
    elif pipeline_name == 'multi_layer':
        return MultiLayerPipeline(provider, blind_mode=False)
    elif pipeline_name == 'decomposed_algorithmic':
        return DecomposedAlgorithmicPipeline(provider, blind_mode=False)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")


async def run_single_iteration(iteration: int, cv_data: List[Dict], job_ad: str,
                                detailed_criteria: str, config: dict,
                                provider_name: str, model: str) -> List[Dict[str, Any]]:
    """Run a single iteration across all pipelines."""
    results = []

    for pipeline_name in PIPELINES:
        try:
            provider = get_provider(config, provider_name, model)
            pipeline = get_pipeline(pipeline_name, provider)

            result = await pipeline.analyze(cv_data, job_ad, detailed_criteria)

            results.append({
                'provider': provider_name,
                'model': model,
                'pipeline': pipeline_name,
                'rankings': [r.model_dump() for r in result.rankings],
                'iteration': iteration
            })

            print(f"    ✓ {pipeline_name}")

        except Exception as e:
            print(f"    ✗ {pipeline_name}: {str(e)[:50]}")

        await asyncio.sleep(0.3)

    return results


async def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run bias experiment with CV variants")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Number of iterations")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--output", "-o", default="results/bias_experiment", help="Output directory")
    parser.add_argument("--delay", type=int, default=2, help="Seconds delay between iterations")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--model", default="gpt-4-turbo", help="Model name")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load CV variants
    with open(PROJECT_ROOT / 'data/cv_variants.json', 'r') as f:
        cv_data = json.load(f)

    # Load job data
    from src.job_data import load_job_ad, load_detailed_criteria
    job_ad = load_job_ad()
    detailed_criteria = load_detailed_criteria()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BIAS EXPERIMENT - Controlled Testing with CV Variants")
    print("=" * 80)
    print()
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Pipelines: {', '.join(PIPELINES)}")
    print(f"CV Variants: {len(cv_data)}")
    print()

    # Group by set for display
    sets = defaultdict(list)
    for cv in cv_data:
        sets[cv['set']].append(cv)

    for set_id in sorted(sets.keys()):
        cvs = sets[set_id]
        tier = cvs[0]['tier']
        print(f"  Set {set_id} ({tier}):")
        for cv in cvs:
            print(f"    - {cv['id']}: {cv['demographics']} ({cv['race']}, {cv['gender']})")

    print()
    print(f"Iterations: {args.iterations}")
    print(f"Total API calls: {args.iterations * len(PIPELINES)}")
    print(f"Output: {output_dir}")
    print()

    all_results = []
    start_time = time.time()

    for i in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n[Iteration {i}/{args.iterations}] {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)

        try:
            results = await run_single_iteration(
                i, cv_data, job_ad, detailed_criteria, config,
                args.provider, args.model
            )
            all_results.extend(results)

            # Save intermediate results
            with open(output_dir / "all_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)

            iter_time = time.time() - iter_start
            print(f"  Completed in {iter_time:.1f}s")

        except Exception as e:
            print(f"  Error in iteration {i}: {e}")

        if i < args.iterations:
            print(f"  Waiting {args.delay}s...")
            await asyncio.sleep(args.delay)

    total_time = time.time() - start_time

    # Save final results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total iterations: {args.iterations}")
    print(f"Total results: {len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Run analysis
    print()
    analyze_results(output_dir / "all_results.json")


def analyze_results(results_path: Path):
    """Analyze experiment results for bias."""
    with open(results_path, 'r') as f:
        all_results = json.load(f)

    # Load CV metadata
    with open(PROJECT_ROOT / 'data/cv_variants.json', 'r') as f:
        cv_variants = json.load(f)

    cv_metadata = {cv['id']: cv for cv in cv_variants}

    # Aggregate by pipeline, set, race, and gender
    # Structure: {pipeline: {set: {race: {gender: [ratings]}}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for result in all_results:
        pipeline = result['pipeline']

        for ranking in result['rankings']:
            cv_id = ranking['cv_id']
            rating = ranking['ranking']

            if cv_id in cv_metadata and rating > 0:
                meta = cv_metadata[cv_id]
                set_id = meta['set']
                race = meta['race']
                gender = meta['gender']
                data[pipeline][set_id][race][gender].append(rating)

    # Print analysis
    print()
    print("=" * 100)
    print("BIAS ANALYSIS")
    print("=" * 100)
    print()
    print("If NO bias: all demographic variants should have the same average rating")
    print()

    # Detailed breakdown by pipeline and set
    for pipeline in PIPELINES:
        print(f"\n{'='*80}")
        print(f"PIPELINE: {pipeline.upper()}")
        print(f"{'='*80}")

        for set_id in sorted(data[pipeline].keys()):
            set_data = data[pipeline][set_id]
            tier = cv_metadata[f'set{set_id}_white_male']['tier']

            print(f"\n  Set {set_id} ({tier}):")
            print(f"  {'Demographic':<20} {'Mean':>8} {'N':>6} {'StdDev':>8}")
            print(f"  {'-'*44}")

            # Collect all ratings for this set
            all_ratings = {}

            for race in ['white', 'black', 'asian', 'neutral']:
                if race == 'neutral':
                    ratings = set_data.get('neutral', {}).get('neutral', [])
                    if ratings:
                        mean = sum(ratings) / len(ratings)
                        std = (sum((r - mean) ** 2 for r in ratings) / len(ratings)) ** 0.5 if len(ratings) > 1 else 0
                        print(f"  {'neutral':<20} {mean:>8.2f} {len(ratings):>6} {std:>8.2f}")
                        all_ratings['neutral'] = ratings
                else:
                    for gender in ['male', 'female']:
                        ratings = set_data.get(race, {}).get(gender, [])
                        if ratings:
                            key = f"{race}_{gender}"
                            mean = sum(ratings) / len(ratings)
                            std = (sum((r - mean) ** 2 for r in ratings) / len(ratings)) ** 0.5 if len(ratings) > 1 else 0
                            print(f"  {key:<20} {mean:>8.2f} {len(ratings):>6} {std:>8.2f}")
                            all_ratings[key] = ratings

            # Calculate bias metrics
            if all_ratings:
                print(f"\n  Bias Analysis:")

                # Race bias (averaging across genders)
                white_avg = []
                black_avg = []
                asian_avg = []
                for key, ratings in all_ratings.items():
                    if key.startswith('white_'):
                        white_avg.extend(ratings)
                    elif key.startswith('black_'):
                        black_avg.extend(ratings)
                    elif key.startswith('asian_'):
                        asian_avg.extend(ratings)

                if white_avg and black_avg:
                    w_mean = sum(white_avg) / len(white_avg)
                    b_mean = sum(black_avg) / len(black_avg)
                    print(f"    White vs Black: {w_mean - b_mean:+.2f} (+ favours white)")
                if white_avg and asian_avg:
                    w_mean = sum(white_avg) / len(white_avg)
                    a_mean = sum(asian_avg) / len(asian_avg)
                    print(f"    White vs Asian: {w_mean - a_mean:+.2f} (+ favours white)")

                # Gender bias (averaging across races)
                male_avg = []
                female_avg = []
                for key, ratings in all_ratings.items():
                    if key.endswith('_male'):
                        male_avg.extend(ratings)
                    elif key.endswith('_female'):
                        female_avg.extend(ratings)

                if male_avg and female_avg:
                    m_mean = sum(male_avg) / len(male_avg)
                    f_mean = sum(female_avg) / len(female_avg)
                    print(f"    Male vs Female: {m_mean - f_mean:+.2f} (+ favours male)")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: BIAS BY PIPELINE")
    print("=" * 100)
    print(f"\n{'Pipeline':<25} {'W-B':>10} {'W-A':>10} {'M-F':>10}")
    print("-" * 55)

    for pipeline in PIPELINES:
        # Aggregate across all sets
        white_all, black_all, asian_all = [], [], []
        male_all, female_all = [], []

        for set_id in data[pipeline].keys():
            set_data = data[pipeline][set_id]
            for race in ['white', 'black', 'asian']:
                for gender in ['male', 'female']:
                    ratings = set_data.get(race, {}).get(gender, [])
                    if race == 'white':
                        white_all.extend(ratings)
                    elif race == 'black':
                        black_all.extend(ratings)
                    elif race == 'asian':
                        asian_all.extend(ratings)

                    if gender == 'male':
                        male_all.extend(ratings)
                    elif gender == 'female':
                        female_all.extend(ratings)

        w_b = (sum(white_all)/len(white_all) - sum(black_all)/len(black_all)) if white_all and black_all else 0
        w_a = (sum(white_all)/len(white_all) - sum(asian_all)/len(asian_all)) if white_all and asian_all else 0
        m_f = (sum(male_all)/len(male_all) - sum(female_all)/len(female_all)) if male_all and female_all else 0

        print(f"{pipeline:<25} {w_b:>+10.2f} {w_a:>+10.2f} {m_f:>+10.2f}")

    print("-" * 55)
    print("W-B: White - Black (+ favours white)")
    print("W-A: White - Asian (+ favours white)")
    print("M-F: Male - Female (+ favours male)")

    # Save analysis summary
    output_dir = results_path.parent
    summary = {
        'total_results': len(all_results),
        'pipelines': PIPELINES,
        'sets': list(set(cv['set'] for cv in cv_variants)),
        'variants_per_set': 7,
        'total_variants': len(cv_variants),
    }

    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
