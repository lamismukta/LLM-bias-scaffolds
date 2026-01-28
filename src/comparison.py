"""Comparison and evaluation framework for pipeline results."""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from .pipelines.base import PipelineResult


class ComparisonFramework:
    """Framework for comparing pipeline results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: List[PipelineResult], experiment_name: str = None):
        """Save pipeline results to disk."""
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_dir = self.results_dir / experiment_name
        
        # Warn if directory already exists (might overwrite)
        if experiment_dir.exists():
            print(f"Warning: Experiment directory '{experiment_name}' already exists.")
            print(f"Results will be saved to: {experiment_dir}")
            print("Consider using a unique experiment name to avoid overwriting.\n")
        
        experiment_dir.mkdir(exist_ok=True)
        
        # Save individual pipeline results
        for result in results:
            filename = f"{result.pipeline_name}_{result.provider}_{result.model}.json"
            filepath = experiment_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump(), f, indent=2)
            
            # Save rankings file (names and rankings)
            rankings_filename = f"{result.pipeline_name}_{result.provider}_{result.model}_rankings.txt"
            rankings_filepath = experiment_dir / rankings_filename
            self._save_rankings_file(result, rankings_filepath)
        
        # Save summary
        summary = self._create_summary(results)
        summary_path = experiment_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Save comparison table
        comparison_df = self.create_comparison_dataframe(results)
        comparison_path = experiment_dir / "comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"Results saved to {experiment_dir}")
        return experiment_dir
    
    def _save_rankings_file(self, result: PipelineResult, filepath: Path):
        """Save a human-readable rankings file with names and rankings."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline: {result.pipeline_name}\n")
            f.write(f"Provider: {result.provider}\n")
            f.write(f"Model: {result.model}\n")
            f.write("=" * 60 + "\n\n")
            
            # Sort by ranking (descending) then by name
            sorted_rankings = sorted(
                result.rankings,
                key=lambda x: (-x.ranking, x.name)
            )
            
            for ranking in sorted_rankings:
                f.write(f"Ranking: {ranking.ranking} ({self._ranking_label(ranking.ranking)})\n")
                f.write(f"Name: {ranking.name}\n")
                f.write(f"CV ID: {ranking.cv_id}\n")
                f.write(f"Reasoning: {ranking.reasoning}\n")
                f.write("-" * 60 + "\n\n")
    
    def _ranking_label(self, ranking: int) -> str:
        """Convert ranking number to label."""
        labels = {4: "Excellent Fit", 3: "Good Fit", 2: "Borderline", 1: "Not a Fit"}
        return labels.get(ranking, "Unknown")
    
    def _create_summary(self, results: List[PipelineResult]) -> Dict[str, Any]:
        """Create a summary of results."""
        summary = {
            "experiment_timestamp": datetime.now().isoformat(),
            "total_pipeline_runs": len(results),
            "pipelines": {},
            "providers": {},
            "models": {},
        }
        
        # Group by pipeline
        for result in results:
            pipeline_key = result.pipeline_name
            if pipeline_key not in summary["pipelines"]:
                summary["pipelines"][pipeline_key] = {
                    "count": 0,
                    "models": set(),
                    "total_tokens": 0,
                    "cv_count": 0
                }
            summary["pipelines"][pipeline_key]["count"] += 1
            summary["pipelines"][pipeline_key]["models"].add(result.model)
            summary["pipelines"][pipeline_key]["cv_count"] = len(result.rankings)
            
            # Get token usage
            usage = result.metadata.get("usage", {})
            if isinstance(usage, dict):
                summary["pipelines"][pipeline_key]["total_tokens"] += usage.get("total_tokens", 0)
        
        # Convert sets to lists for JSON serialization
        for pipeline in summary["pipelines"].values():
            pipeline["models"] = list(pipeline["models"])
        
        return summary
    
    def create_comparison_dataframe(self, results: List[PipelineResult]) -> pd.DataFrame:
        """Create a pandas DataFrame for easy comparison."""
        rows = []
        for result in results:
            for ranking in result.rankings:
                row = {
                    "cv_id": ranking.cv_id,
                    "name": ranking.name,
                    "pipeline": result.pipeline_name,
                    "provider": result.provider,
                    "model": result.model,
                    "ranking": ranking.ranking,
                    "ranking_label": self._ranking_label(ranking.ranking),
                    "reasoning": ranking.reasoning,
                }
                
                # Add token usage (same for all CVs in this pipeline run)
                usage = result.metadata.get("usage", {})
                if isinstance(usage, dict):
                    row["total_tokens"] = usage.get("total_tokens", 0)
                    row["prompt_tokens"] = usage.get("prompt_tokens", 0)
                    row["completion_tokens"] = usage.get("completion_tokens", 0)
                else:
                    row["total_tokens"] = 0
                    row["prompt_tokens"] = 0
                    row["completion_tokens"] = 0
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def compare_pipelines(self, results: List[PipelineResult], cv_id: str = None) -> pd.DataFrame:
        """Compare results across pipelines for a specific CV or all CVs."""
        comparison_data = []
        for result in results:
            for ranking in result.rankings:
                if cv_id is None or ranking.cv_id == cv_id:
                    comparison_data.append({
                        "cv_id": ranking.cv_id,
                        "name": ranking.name,
                        "pipeline": result.pipeline_name,
                        "model": result.model,
                        "ranking": ranking.ranking,
                        "ranking_label": self._ranking_label(ranking.ranking),
                        "reasoning": ranking.reasoning,
                        "tokens_used": result.metadata.get("usage", {}).get("total_tokens", 0) if isinstance(result.metadata.get("usage"), dict) else 0
                    })
        
        return pd.DataFrame(comparison_data)
    
    def load_results(self, experiment_name: str) -> List[PipelineResult]:
        """Load results from a previous experiment."""
        experiment_dir = self.results_dir / experiment_name
        if not experiment_dir.exists():
            raise ValueError(f"Experiment {experiment_name} not found")
        
        results = []
        for filepath in experiment_dir.glob("*.json"):
            if filepath.name == "summary.json":
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(PipelineResult(**data))
        
        return results
