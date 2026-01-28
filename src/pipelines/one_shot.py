"""One-shot prompt pipeline - single comprehensive analysis."""
import json
import re
import asyncio
from typing import Dict, Any, List
from .base import Pipeline, PipelineResult, RankingResult


class OneShotPipeline(Pipeline):
    """Single prompt analysis pipeline."""

    def __init__(self, llm_provider, blind_mode: bool = False):
        super().__init__(llm_provider, "one_shot", blind_mode)

    def _parse_response(self, response_content: str) -> tuple[int, str, bool]:
        """Parse the LLM response to extract ranking and reasoning.
        Returns (ranking, reasoning, success)."""
        ranking = 0
        reasoning = ""

        # Try JSON parsing first
        parsed = self.extract_json_from_response(response_content)
        if parsed:
            ranking = self.extract_ranking_from_parsed(parsed)
            reasoning_raw = parsed.get("reasoning", parsed.get("justification", ""))
            if isinstance(reasoning_raw, dict):
                reasoning = json.dumps(reasoning_raw, indent=2)
            elif isinstance(reasoning_raw, (list, tuple)):
                reasoning = "\n".join(str(item) for item in reasoning_raw)
            else:
                reasoning = str(reasoning_raw) if reasoning_raw else ""

        # Fallback: try regex patterns if JSON parsing failed or ranking is 0
        if ranking == 0:
            patterns = [
                r'"ranking"\s*:\s*(\d+)',
                r'"overall_fit_rating"\s*:\s*(\d+)',
                r'"fit_rating"\s*:\s*(\d+)',
                r'"rating"\s*:\s*(\d+)',
                r'"score"\s*:\s*(\d+)',
                r'ranking["\s:]+(\d+)',
                r'rating["\s:]+(\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, response_content, re.IGNORECASE)
                if match:
                    val = int(match.group(1))
                    if 1 <= val <= 4:
                        ranking = val
                        break

            if not reasoning:
                reasoning = response_content

        success = ranking > 0
        return ranking, reasoning, success

    async def _analyze_single_cv(self, cv: Dict[str, Any], job_ad: str, detailed_criteria: str, max_retries: int = 2) -> RankingResult:
        """Analyze a single CV independently with retry logic for parsing failures."""
        # Apply blind mode if enabled
        cv = self.prepare_cv(cv)

        prompt = f"""You are evaluating a candidate for a Founding Operator role.

Job Description:
{job_ad}

Detailed Hiring Criteria:
{detailed_criteria}

You will be evaluating this candidate against three key criteria:
1. Zero-to-One Operator
2. Technical T-Shape
3. Recruitment Mastery

Candidate CV:
{cv['content']}

Provide an overall fit rating from 1-4 of the candidate to the role:
- 4 = Excellent fit
- 3 = Good fit
- 2 = Borderline fit
- 1 = Not a fit

IMPORTANT: Respond with ONLY valid JSON in this exact format (no other text):
{{
    "cv_id": "{cv['id']}",
    "ranking": <number 1-4>
}}"""

        # Extract name from original CV content (before anonymization)
        name = self.extract_name_from_cv(cv.get("content", ""))
        if self.blind_mode:
            name = "[BLIND]"

        # Try up to max_retries + 1 times
        ranking = 0
        reasoning = ""
        attempts = 0

        while ranking == 0 and attempts <= max_retries:
            if attempts > 0:
                print(f"    Retry {attempts}/{max_retries} for {cv['id']} (parsing failed)")
                await asyncio.sleep(0.5)

            response = await self.llm_provider.generate(prompt)
            ranking, reasoning, _ = self._parse_response(response.content)
            attempts += 1

        if ranking == 0:
            print(f"    Warning: Failed to parse ranking for {cv['id']} after {attempts} attempts")

        return RankingResult(
            cv_id=cv['id'],
            name=name,
            ranking=ranking,
            reasoning=reasoning
        )

    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Perform one-shot analysis of all CVs - one API call per CV."""

        # Process each CV independently in parallel
        tasks = [self._analyze_single_cv(cv, job_ad, detailed_criteria) for cv in cv_list]
        rankings = await asyncio.gather(*tasks)

        analysis = {
            "note": "Each CV evaluated independently in separate API calls",
            "total_cvs": len(cv_list),
            "blind_mode": self.blind_mode
        }

        return PipelineResult(
            pipeline_name=self.name,
            provider=self.llm_provider.get_provider_name(),
            model=self.llm_provider.model,
            rankings=list(rankings),
            analysis=analysis,
            metadata={
                "usage": {"note": "Token usage not tracked per individual CV call"},
            }
        )
