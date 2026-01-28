"""Base class for analysis pipelines."""
import re
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel
from ..providers.base import LLMProvider


class RankingResult(BaseModel):
    """Result for a single CV ranking."""
    cv_id: str
    name: str
    ranking: int  # 1-4 (4=excellent, 3=good, 2=borderline, 1=not a fit)
    reasoning: str = ""


class PipelineResult(BaseModel):
    """Result from a pipeline execution."""
    pipeline_name: str
    provider: str
    model: str
    rankings: List[RankingResult]  # List of rankings for all CVs
    analysis: Dict[str, Any] = {}  # Additional analysis data
    metadata: Dict[str, Any] = {}


class Pipeline(ABC):
    """Abstract base class for CV analysis pipelines."""

    def __init__(self, llm_provider: LLMProvider, name: str, blind_mode: bool = False):
        self.llm_provider = llm_provider
        self.name = name
        self.blind_mode = blind_mode
        if blind_mode:
            self.name = f"{name}_blind"

    def anonymize_cv(self, cv_content: str) -> str:
        """Remove names and other identifying information from CV content."""
        lines = cv_content.split('\n')
        anonymized_lines = []

        for i, line in enumerate(lines):
            # Skip the first line (usually the name as header)
            if i == 0 and line.strip().startswith('#'):
                anonymized_lines.append("# [CANDIDATE]")
                continue

            # Also anonymize any line that looks like just a name (title case, short)
            stripped = line.strip().replace('#', '').replace('_', '').strip()
            if i < 3 and stripped and len(stripped.split()) <= 4 and stripped.istitle():
                # Likely a name line
                anonymized_lines.append(line.replace(stripped, "[CANDIDATE]"))
            else:
                anonymized_lines.append(line)

        return '\n'.join(anonymized_lines)

    def prepare_cv(self, cv: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare CV for analysis, applying blind mode if enabled."""
        if self.blind_mode:
            return {
                'id': cv['id'],
                'content': self.anonymize_cv(cv.get('content', ''))
            }
        return cv

    async def generate_with_retry(self, prompt: str, max_retries: int = 2,
                                   parse_func: Optional[Callable] = None) -> Any:
        """Generate LLM response with retry logic for parsing failures.

        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts
            parse_func: Optional function to parse response. If provided and returns None/0,
                       will trigger a retry. Should return (parsed_result, success_bool)
        """
        attempts = 0
        last_response = None

        while attempts <= max_retries:
            if attempts > 0:
                await asyncio.sleep(0.5)  # Small delay between retries

            response = await self.llm_provider.generate(prompt)
            last_response = response

            if parse_func is None:
                return response

            parsed, success = parse_func(response.content)
            if success:
                return response

            attempts += 1

        return last_response

    def extract_json_from_response(self, content: str) -> Optional[dict]:
        """Extract JSON from response content, handling markdown code blocks."""
        try:
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return None

    def extract_ranking_from_parsed(self, parsed: dict) -> int:
        """Extract ranking from parsed JSON, checking multiple possible field names."""
        if not isinstance(parsed, dict):
            return 0

        ranking_fields = [
            "ranking", "overall_fit_rating", "fit_rating", "overall_rating",
            "score", "overall_score", "fit_score", "rating"
        ]

        for field in ranking_fields:
            if field in parsed:
                ranking_value = parsed[field]
                if isinstance(ranking_value, int) and 1 <= ranking_value <= 4:
                    return ranking_value
                elif isinstance(ranking_value, (float, str)):
                    try:
                        val = int(float(ranking_value))
                        if 1 <= val <= 4:
                            return val
                    except (ValueError, TypeError):
                        pass

        # Check nested structures
        nested_keys = ["result", "evaluation", "assessment", "analysis"]
        for key in nested_keys:
            if key in parsed and isinstance(parsed[key], dict):
                nested_ranking = self.extract_ranking_from_parsed(parsed[key])
                if nested_ranking > 0:
                    return nested_ranking

        return 0

    def extract_name_from_cv(self, cv_content: str) -> str:
        """Extract candidate name from CV content."""
        if not cv_content:
            return "Unknown"
        first_line = cv_content.split('\n')[0].strip()
        name = first_line.replace('#', '').replace('_', '').strip()
        return name if name else "Unknown"

    @abstractmethod
    async def analyze(self, cv_list: List[Dict[str, Any]], job_ad: str, detailed_criteria: str) -> PipelineResult:
        """Analyze a list of CVs and return rankings.

        Args:
            cv_list: List of dictionaries containing CV information (id, content)
            job_ad: Job advertisement text
            detailed_criteria: Detailed hiring criteria

        Returns:
            PipelineResult with rankings for all CVs
        """
        pass

