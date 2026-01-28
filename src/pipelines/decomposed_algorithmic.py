"""Decomposed pipeline with algorithmic aggregation - shares criteria evaluation with multi_layer."""
import json
import asyncio
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class DecomposedAlgorithmicPipeline(Pipeline):
    """Evaluates criteria separately (like multi_layer) but uses algorithmic aggregation instead of LLM synthesis."""

    def __init__(self, llm_provider, blind_mode: bool = False):
        super().__init__(llm_provider, "decomposed_algorithmic", blind_mode)

    def _map_rating_to_score(self, rating: str) -> int:
        """Map qualitative rating to numeric score."""
        rating_lower = rating.lower().strip()

        if 'excellent' in rating_lower:
            return 4
        elif 'good' in rating_lower:
            return 3
        elif 'weak' in rating_lower or 'borderline' in rating_lower:
            return 2
        elif 'not a fit' in rating_lower or 'not fit' in rating_lower:
            return 1
        else:
            # Default to borderline if unclear
            return 2

    def _aggregate_scores(self, criteria_evaluations: Dict[str, Any]) -> tuple[int, str]:
        """Aggregate criteria scores using simple average (no weights)."""

        scores = []
        reasoning_parts = []

        for criteria_key in ['zero_to_one', 'technical_t_shape', 'recruitment_mastery']:
            eval_data = criteria_evaluations.get(criteria_key, {})

            if isinstance(eval_data, dict):
                rating = eval_data.get('rating', 'Unknown')
                score = self._map_rating_to_score(rating)
                scores.append(score)
                reasoning_parts.append(f"{criteria_key}: {rating} (score: {score})")
            else:
                # Error case
                scores.append(2)  # Default to borderline
                reasoning_parts.append(f"{criteria_key}: Error in evaluation")

        if not scores:
            return 2, "No valid criteria evaluations"

        # Simple average (rounded to nearest integer)
        avg_score = sum(scores) / len(scores)
        final_ranking = round(avg_score)

        # Ensure ranking is in valid range
        final_ranking = max(1, min(4, final_ranking))

        reasoning = (
            f"Algorithmic aggregation: Average of criteria scores = {avg_score:.2f} → {final_ranking}\n"
            + "\n".join(reasoning_parts)
        )

        return final_ranking, reasoning

    def _extract_criteria_section(self, detailed_criteria: str, criteria_name: str) -> str:
        """Extract the relevant section from detailed criteria."""
        lines = detailed_criteria.split('\n')
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if criteria_name.lower() in line.lower() and '#' in line:
                start_idx = i
            elif start_idx is not None and line.strip().startswith('#') and criteria_name.lower() not in line.lower():
                end_idx = i
                break

        if start_idx is not None:
            if end_idx is None:
                end_idx = len(lines)
            return '\n'.join(lines[start_idx:end_idx])

        return detailed_criteria  # Fallback to full criteria

    async def _evaluate_single_criteria(self, cv: Dict[str, Any], job_ad: str,
                                         criteria_name: str, criteria_key: str,
                                         criteria_section: str, max_retries: int = 2) -> Dict[str, Any]:
        """Evaluate a single criterion with retry logic."""
        prompt = f"""Evaluate this candidate against the "{criteria_name}" criteria.

Job Description:
{job_ad}

Criteria Details:
{criteria_section}

Candidate CV:
{cv['content']}

Evaluate their fit to this specific criteria and rate as: Excellent, Good, Weak, or Not a Fit.

Provide your evaluation in JSON format:
{{
    "cv_id": "{cv['id']}",
    "rating": "Excellent/Good/Weak/Not a Fit",
    "evidence": "Specific evidence from CV supporting this rating"
}}"""

        attempts = 0
        response = None
        while attempts <= max_retries:
            if attempts > 0:
                await asyncio.sleep(0.5)

            response = await self.llm_provider.generate(prompt)

            try:
                parsed = self.extract_json_from_response(response.content)
                if parsed and "rating" in parsed:
                    return {
                        "cv_id": cv['id'],
                        "rating": parsed.get("rating", "Unknown"),
                        "evidence": parsed.get("evidence", "")
                    }
            except Exception:
                pass

            attempts += 1

        # Return error result after all retries
        return {
            "cv_id": cv['id'],
            "error": "Failed to parse after retries",
            "raw": response.content if response else "",
            "rating": "Unknown"
        }

    async def _analyze_single_cv(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str, max_retries: int = 2) -> tuple:
        """Analyze a single CV with decomposed criteria evaluation and algorithmic aggregation."""
        # Apply blind mode if enabled
        cv = self.prepare_cv(cv)

        # Layer 1: Evaluate each criteria separately in PARALLEL
        criteria_list = [
            ("Zero-to-One Operator", "zero_to_one"),
            ("Technical T-Shape", "technical_t_shape"),
            ("Recruitment Mastery", "recruitment_mastery")
        ]

        # Create tasks for parallel criteria evaluation
        criteria_tasks = []
        for criteria_name, criteria_key in criteria_list:
            criteria_section = self._extract_criteria_section(detailed_criteria, criteria_name)
            task = self._evaluate_single_criteria(cv, job_ad, criteria_name, criteria_key, criteria_section, max_retries)
            criteria_tasks.append((criteria_key, task))

        # Run all criteria evaluations in parallel
        results = await asyncio.gather(*[task for _, task in criteria_tasks])
        criteria_evaluations = {criteria_tasks[i][0]: results[i] for i in range(len(results))}

        # Layer 2: Algorithmic aggregation (no API call)
        final_ranking, reasoning = self._aggregate_scores(criteria_evaluations)

        # Extract name from CV content
        name = self.extract_name_from_cv(cv.get("content", ""))
        if self.blind_mode:
            name = "[BLIND]"

        ranking_result = RankingResult(
            cv_id=cv['id'],
            name=name,
            ranking=final_ranking,
            reasoning=reasoning
        )

        return ranking_result, criteria_evaluations

    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform decomposed analysis with algorithmic aggregation - CVs processed in parallel."""

        # Process each CV independently in PARALLEL
        tasks = [self._analyze_single_cv(cv, job_ad, detailed_criteria) for cv in cv_list]
        results = await asyncio.gather(*tasks)

        rankings = []
        all_criteria_evals = {}

        for ranking_result, criteria_evaluations in results:
            rankings.append(ranking_result)
            all_criteria_evals[ranking_result.cv_id] = criteria_evaluations

        analysis = {
            "note": "Criteria evaluated via LLM (3 API calls per CV in parallel), aggregated algorithmically (simple average)",
            "total_cvs": len(cv_list),
            "blind_mode": self.blind_mode,
            "criteria_evaluations": all_criteria_evals,
            "aggregation_method": "Simple average of criteria scores (no weights)"
        }

        return PipelineResult(
            pipeline_name=self.name,
            provider=self.llm_provider.get_provider_name(),
            model=self.llm_provider.model,
            rankings=rankings,
            analysis=analysis,
            metadata={
                "usage": {"note": "Token usage not tracked per individual CV call"},
            }
        )
