"""Load job ad and criteria data."""
import json
from pathlib import Path


def load_job_ad() -> str:
    """Load job ad from Python file."""
    job_ad_path = Path(__file__).parent.parent / "data" / "jobAd.py"
    
    # Read and execute the Python file to get the variables
    job_ad = None
    with open(job_ad_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Create a local namespace to execute the file
        namespace = {}
        exec(content, namespace)
        job_ad = namespace.get('job_ad', '')
    
    return job_ad.strip() if job_ad else ''


def load_detailed_criteria() -> str:
    """Load detailed hiring criteria."""
    job_ad_path = Path(__file__).parent.parent / "data" / "jobAd.py"
    
    detailed_hiring_criteria = None
    with open(job_ad_path, 'r', encoding='utf-8') as f:
        content = f.read()
        namespace = {}
        exec(content, namespace)
        detailed_hiring_criteria = namespace.get('detailed_hiring_criteria', '')
    
    return detailed_hiring_criteria.strip() if detailed_hiring_criteria else ''


def load_category_guidance() -> list:
    """Load category guidance from JSON."""
    guidance_path = Path(__file__).parent.parent / "data" / "category_guidance.json"
    with open(guidance_path, 'r') as f:
        return json.load(f)

