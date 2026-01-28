#!/usr/bin/env python3
"""Example script showing how to use the pipeline framework programmatically."""
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from src.providers.openai_provider import OpenAIProvider
from src.pipelines.one_shot import OneShotPipeline
from src.pipelines.chain_of_thought import ChainOfThoughtPipeline
from src.pipelines.multi_layer import MultiLayerPipeline
from src.comparison import ComparisonFramework


async def example_single_cv():
    """Example: Analyze a single CV with one pipeline."""
    load_dotenv()
    
    # Load CV data
    with open("data/cvs_revised_v2.json", 'r') as f:
        cvs = json.load(f)
    
    # Get first CV
    cv = cvs[0]
    print(f"Analyzing CV: {cv['name']} ({cv['id']})")
    
    # Create provider
    provider = OpenAIProvider(model="gpt-4o-mini")
    
    # Create pipeline
    pipeline = OneShotPipeline(provider)
    
    # Run analysis
    result = await pipeline.analyze(cv)
    
    print(f"\nPipeline: {result.pipeline_name}")
    print(f"Model: {result.model}")
    print(f"Rating: {result.analysis.get('overall_rating', 'N/A')}")
    print(f"Tokens used: {result.metadata.get('usage', {}).get('total_tokens', 0)}")
    print(f"\nAnalysis:\n{json.dumps(result.analysis, indent=2)}")


async def example_compare_pipelines():
    """Example: Compare different pipelines on the same CV."""
    load_dotenv()
    
    # Load CV data
    with open("data/cvs_revised_v2.json", 'r') as f:
        cvs = json.load(f)
    
    cv = cvs[0]
    print(f"Comparing pipelines on CV: {cv['name']} ({cv['id']})\n")
    
    # Create provider
    provider = OpenAIProvider(model="gpt-4o-mini")
    
    # Create all pipelines
    pipelines = [
        OneShotPipeline(provider),
        ChainOfThoughtPipeline(provider),
        MultiLayerPipeline(provider)
    ]
    
    # Run all pipelines
    results = []
    for pipeline in pipelines:
        result = await pipeline.analyze(cv)
        results.append(result)
        print(f"{pipeline.name}: {result.analysis.get('overall_rating', 'N/A')} "
              f"({result.metadata.get('usage', {}).get('total_tokens', 0)} tokens)")
    
    # Compare
    framework = ComparisonFramework()
    comparison_df = framework.compare_pipelines(results, cv_id=cv['id'])
    print("\nComparison DataFrame:")
    print(comparison_df[['pipeline', 'rating', 'tokens_used']])


async def example_custom_experiment():
    """Example: Run a custom experiment and save results."""
    load_dotenv()
    
    # Load CV data
    with open("data/cvs_revised_v2.json", 'r') as f:
        cvs = json.load(f)
    
    # Test on first 3 CVs
    test_cvs = cvs[:3]
    
    # Create provider
    provider = OpenAIProvider(model="gpt-4o-mini")
    
    # Create pipelines
    one_shot = OneShotPipeline(provider)
    cot = ChainOfThoughtPipeline(provider)
    
    # Run experiment
    results = []
    for cv in test_cvs:
        print(f"Processing {cv['id']}...")
        results.append(await one_shot.analyze(cv))
        results.append(await cot.analyze(cv))
    
    # Save results
    framework = ComparisonFramework()
    experiment_dir = framework.save_results(results, experiment_name="custom_example")
    
    print(f"\nResults saved to: {experiment_dir}")
    
    # Create comparison
    comparison_df = framework.create_comparison_dataframe(results)
    print("\nComparison Summary:")
    print(comparison_df.groupby(['pipeline', 'overall_rating']).size())


if __name__ == "__main__":
    print("="*60)
    print("Example 1: Single CV Analysis")
    print("="*60)
    asyncio.run(example_single_cv())
    
    print("\n" + "="*60)
    print("Example 2: Compare Pipelines")
    print("="*60)
    asyncio.run(example_compare_pipelines())
    
    print("\n" + "="*60)
    print("Example 3: Custom Experiment")
    print("="*60)
    asyncio.run(example_custom_experiment())

