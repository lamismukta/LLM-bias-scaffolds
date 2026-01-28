"""Multi-layer pipeline - iterative refinement approach."""
import json
import re
import asyncio
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class MultiLayerPipeline(Pipeline):
    """Multi-layer analysis pipeline with iterative refinement."""

    def __init__(self, llm_provider, blind_mode: bool = False):
        super().__init__(llm_provider, "multi_layer", blind_mode)

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
    "rating": "Excellent/Good/Weak/Not a Fit"
}}"""

        attempts = 0
        while attempts <= max_retries:
            if attempts > 0:
                await asyncio.sleep(0.5)

            response = await self.llm_provider.generate(prompt)

            try:
                parsed = self.extract_json_from_response(response.content)
                if parsed and "rating" in parsed:
                    return {
                        "cv_id": cv['id'],
                        "rating": parsed.get("rating", "Unknown")
                    }
            except Exception:
                pass

            attempts += 1

        # Return error result after all retries
        return {
            "cv_id": cv['id'],
            "error": "Failed to parse after retries",
            "raw": response.content if 'response' in dir() else "",
            "rating": "Unknown"
        }

    async def _analyze_single_cv(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str, max_retries: int = 2) -> tuple:
        """Analyze a single CV independently with multi-layer approach."""
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

        # Layer 2: Synthesize overall fit based on criteria evaluations
        synthesis_prompt = f"""Based on the individual criteria evaluations below, determine the overall fit rating (1-4) for this candidate.

Job Description:
{job_ad}

Individual Criteria Evaluations:
{json.dumps(criteria_evaluations, indent=2)}

Synthesize the three criteria evaluations into an overall fit rating:
- 4 = Excellent fit (meets all criteria at excellent level)
- 3 = Good fit (meets criteria at good level)
- 2 = Borderline fit (meets some criteria but has gaps)
- 1 = Not a fit (does not meet key criteria)

Provide your final ranking in JSON format:
{{
    "cv_id": "{cv['id']}",
    "criteria_evaluations": {{
        "zero_to_one": "{criteria_evaluations.get('zero_to_one', {}).get('rating', 'Unknown')}",
        "technical_t_shape": "{criteria_evaluations.get('technical_t_shape', {}).get('rating', 'Unknown')}",
        "recruitment_mastery": "{criteria_evaluations.get('recruitment_mastery', {}).get('rating', 'Unknown')}"
    }},
    "ranking": 4
}}"""

        # Extract name from CV content
        name = self.extract_name_from_cv(cv.get("content", ""))
        if self.blind_mode:
            name = "[BLIND]"

        # Synthesis with retry logic
        ranking = 0
        reasoning = ""
        criteria_eval_summary = {}
        attempts = 0

        while ranking == 0 and attempts <= max_retries:
            if attempts > 0:
                print(f"    Retry {attempts}/{max_retries} for {cv['id']} synthesis (parsing failed)")
                await asyncio.sleep(0.5)

            synthesis_response = await self.llm_provider.generate(synthesis_prompt)

            try:
                parsed = self.extract_json_from_response(synthesis_response.content)
                if parsed and isinstance(parsed, dict):
                    ranking = self.extract_ranking_from_parsed(parsed)
                    reasoning_raw = parsed.get("reasoning", "")
                    criteria_eval_summary = parsed.get("criteria_evaluations", {})

                    if isinstance(reasoning_raw, dict):
                        reasoning = json.dumps(reasoning_raw, indent=2)
                    elif isinstance(reasoning_raw, (list, tuple)):
                        reasoning = "\n".join(str(item) for item in reasoning_raw)
                    else:
                        reasoning = str(reasoning_raw) if reasoning_raw else ""
            except Exception:
                pass

            # Fallback: try regex
            if ranking == 0:
                match = re.search(r'"ranking"\s*:\s*(\d+)', synthesis_response.content)
                if match:
                    val = int(match.group(1))
                    if 1 <= val <= 4:
                        ranking = val
                reasoning = synthesis_response.content

            attempts += 1

        if ranking == 0:
            print(f"    Warning: Failed to parse synthesis for {cv['id']} after {attempts} attempts")

        ranking_result = RankingResult(
            cv_id=cv['id'],
            name=name,
            ranking=ranking,
            reasoning=reasoning
        )

        analysis_data = {
            "layer_1_criteria_evaluations": criteria_evaluations,
            "layer_2_synthesis": {
                "criteria_evaluations": criteria_eval_summary,
                "ranking": ranking
            }
        }

        return ranking_result, analysis_data

    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform multi-layer analysis - each CV evaluated independently in parallel."""

        # Process each CV independently in parallel
        tasks = [self._analyze_single_cv(cv, job_ad, detailed_criteria) for cv in cv_list]
        results = await asyncio.gather(*tasks)

        # Separate rankings and analysis
        rankings = []
        all_analysis = {}
        for ranking_result, analysis_data in results:
            rankings.append(ranking_result)
            all_analysis[ranking_result.cv_id] = analysis_data

        analysis = {
            "note": "Each CV evaluated independently - 4 API calls per CV (3 criteria + 1 synthesis)",
            "total_cvs": len(cv_list),
            "blind_mode": self.blind_mode,
            "per_cv_analyses": all_analysis
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
