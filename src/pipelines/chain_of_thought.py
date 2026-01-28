"""Chain-of-thought pipeline - step-by-step reasoning."""
import json
import re
import asyncio
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class ChainOfThoughtPipeline(Pipeline):
    """Chain-of-thought reasoning pipeline with explicit steps."""

    def __init__(self, llm_provider, blind_mode: bool = False):
        super().__init__(llm_provider, "chain_of_thought", blind_mode)

    def _parse_response(self, response_content: str) -> tuple[int, str, dict, bool]:
        """Parse the LLM response to extract ranking, reasoning, and step analysis.
        Returns (ranking, reasoning, step_analysis, success)."""
        ranking = 0
        reasoning = ""
        step_analysis = {}

        # Try JSON parsing first
        parsed = self.extract_json_from_response(response_content)
        if parsed and isinstance(parsed, dict):
            ranking = self.extract_ranking_from_parsed(parsed)
            reasoning_raw = parsed.get("reasoning", "")
            step_analysis = parsed.get("step_by_step_analysis", {})

            if isinstance(reasoning_raw, dict):
                reasoning = json.dumps(reasoning_raw, indent=2)
            elif isinstance(reasoning_raw, (list, tuple)):
                reasoning = "\n".join(str(item) for item in reasoning_raw)
            else:
                reasoning = str(reasoning_raw) if reasoning_raw else ""

        # Fallback: try regex patterns
        if ranking == 0:
            patterns = [
                r'"ranking"\s*:\s*(\d+)',
                r'"overall_fit_rating"\s*:\s*(\d+)',
                r'"rating"\s*:\s*(\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, response_content, re.IGNORECASE)
                if match:
                    val = int(match.group(1))
                    if 1 <= val <= 4:
                        ranking = val
                        break
            reasoning = response_content

        success = ranking > 0
        return ranking, reasoning, step_analysis, success

    async def _analyze_single_cv(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str, max_retries: int = 2) -> tuple:
        """Analyze a single CV independently with chain-of-thought and retry logic."""
        # Apply blind mode if enabled
        cv = self.prepare_cv(cv)

        prompt = f"""You are evaluating a candidate for a Founding Operator role using a step-by-step chain-of-thought approach.

Job Description:
{job_ad}

Detailed Hiring Criteria:
{detailed_criteria}

Candidate CV:
{cv['content']}

Follow this step-by-step process:

Step 1: Evaluate Zero-to-One Operator fit
- Assess their experience building operational systems from scratch
- Look for evidence of "diagnose, build, then scale" mindset
- Rate: Excellent / Good / Weak / Not a Fit

Step 2: Evaluate Technical T-Shape fit
- Assess technical/analytical depth and ability to partner with engineers
- Look for evidence of AI tool usage and automation experience
- Rate: Excellent / Good / Weak / Not a Fit

Step 3: Evaluate Recruitment Mastery fit
- Assess end-to-end recruitment experience
- Look for evidence of building hiring pipelines from scratch
- Rate: Excellent / Good / Weak / Not a Fit

Step 4: Synthesize overall fit
- Consider all three criteria together
- Determine overall rating: 4 (Excellent), 3 (Good), 2 (Borderline), 1 (Not a Fit)

After completing your step-by-step analysis, provide your final ranking in JSON format:
{{
    "cv_id": "{cv['id']}",
    "step_by_step_analysis": {{
        "zero_to_one": "Excellent/Good/Weak/Not a Fit - reasoning",
        "technical_t_shape": "Excellent/Good/Weak/Not a Fit - reasoning",
        "recruitment_mastery": "Excellent/Good/Weak/Not a Fit - reasoning",
        "synthesis": "Overall reasoning"
    }},
    "ranking": 4
}}"""

        # Extract name from CV content
        name = self.extract_name_from_cv(cv.get("content", ""))
        if self.blind_mode:
            name = "[BLIND]"

        # Try up to max_retries + 1 times
        ranking = 0
        reasoning = ""
        step_analysis = {}
        attempts = 0

        while ranking == 0 and attempts <= max_retries:
            if attempts > 0:
                print(f"    Retry {attempts}/{max_retries} for {cv['id']} (parsing failed)")
                await asyncio.sleep(0.5)

            response = await self.llm_provider.generate(prompt)
            ranking, reasoning, step_analysis, _ = self._parse_response(response.content)
            attempts += 1

        if ranking == 0:
            print(f"    Warning: Failed to parse ranking for {cv['id']} after {attempts} attempts")

        ranking_result = RankingResult(
            cv_id=cv['id'],
            name=name,
            ranking=ranking,
            reasoning=reasoning
        )
        return ranking_result, step_analysis

    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform chain-of-thought analysis - one API call per CV."""

        # Process each CV independently in parallel
        tasks = [self._analyze_single_cv(cv, job_ad, detailed_criteria) for cv in cv_list]
        results = await asyncio.gather(*tasks)

        # Separate rankings and analysis
        rankings = []
        all_analysis = {}
        for ranking_result, step_analysis in results:
            rankings.append(ranking_result)
            all_analysis[ranking_result.cv_id] = step_analysis

        analysis = {
            "note": "Each CV evaluated independently in separate API calls",
            "total_cvs": len(cv_list),
            "blind_mode": self.blind_mode,
            "step_by_step_analyses": all_analysis
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
