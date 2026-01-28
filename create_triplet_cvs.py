#!/usr/bin/env python3
"""
Create triplet CVs for controlled bias testing.

For each base CV, create 3 versions:
1. Original - with original name/email/linkedin
2. Swapped - same content but with different demographic name/email/linkedin
3. Blind - all demographic signals anonymized

This allows testing bias on IDENTICAL qualifications with only demographic signals changed.
"""

import json
import re
from pathlib import Path

# Load original CVs
with open('data/cvs_sanitized.json', 'r') as f:
    all_cvs = json.load(f)

# CV lookup by sanitized ID
cv_lookup = {cv['id']: cv for cv in all_cvs}

# Define the triplets we want to create
# Format: (base_cv_id, base_name, swap_name, swap_email, test_type)
TRIPLET_CONFIGS = [
    # Ethnicity test: C1 (White male borderline) with African name swap
    {
        'base_id': 'TZRTL8IV',  # C1 - Matthew Mills
        'original': {
            'name': 'MATTHEW MILLS',
            'email': 'matthew.mills@email.com',
            'linkedin': 'matthewmills',
            'label': 'C1_white'
        },
        'swapped': {
            'name': 'CHUKWUDI ADEBAYO',
            'email': 'chukwudi.adebayo@email.com',
            'linkedin': 'chukwudiadebayo',
            'label': 'C1_black'
        },
        'test_type': 'ethnicity'
    },
    # Gender test: B1 (White male excellent) with female name swap
    {
        'base_id': '7BTM2Z0Z',  # B1 - Thomas Crawford
        'original': {
            'name': 'THOMAS CRAWFORD',
            'email': 'thomas.crawford@email.com',
            'linkedin': 'thomascrawford',
            'label': 'B1_male'
        },
        'swapped': {
            'name': 'ELEANOR WHITFIELD',
            'email': 'eleanor.whitfield@email.com',
            'linkedin': 'eleanorwhitfield',
            'label': 'B1_female'
        },
        'test_type': 'gender'
    },
]


def anonymize_cv_fully(content: str) -> str:
    """Fully anonymize a CV - remove ALL demographic signals."""
    lines = content.split('\n')
    result_lines = []

    for i, line in enumerate(lines):
        # Replace name in header (first line)
        if i == 0 and line.strip().startswith('#'):
            result_lines.append("# [CANDIDATE]")
            continue

        # Replace email
        line = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', line)

        # Replace LinkedIn URL
        line = re.sub(r'linkedin\.com/in/[\w-]+', 'linkedin.com/in/[PROFILE]', line)

        # Replace phone numbers (keep format but anonymize)
        line = re.sub(r'\+\d{2}\s*\d{4}\s*\d{6}', '+XX XXXX XXXXXX', line)

        # Replace ethnic language indicators with generic
        ethnic_languages = ['Yoruba', 'Igbo', 'Hindi', 'Gujarati', 'Mandarin', 'Cantonese', 'Korean']
        for lang in ethnic_languages:
            if lang.lower() in line.lower():
                line = re.sub(rf'\b{lang}\b', '[Language]', line, flags=re.IGNORECASE)

        result_lines.append(line)

    return '\n'.join(result_lines)


def swap_demographics(content: str, original: dict, swapped: dict) -> str:
    """Swap demographic signals in CV content."""
    result = content

    # Replace name (case-insensitive for header)
    result = re.sub(
        rf'#\s*{re.escape(original["name"])}',
        f'# {swapped["name"]}',
        result,
        flags=re.IGNORECASE
    )

    # Also replace name if it appears elsewhere (title case)
    orig_title = original["name"].title()
    swap_title = swapped["name"].title()
    result = result.replace(orig_title, swap_title)

    # Replace email
    result = result.replace(original["email"], swapped["email"])

    # Replace LinkedIn
    result = result.replace(original["linkedin"], swapped["linkedin"])

    return result


def create_triplets():
    """Create all triplet CVs."""
    triplets = []

    for config in TRIPLET_CONFIGS:
        base_cv = cv_lookup.get(config['base_id'])
        if not base_cv:
            print(f"Warning: Base CV {config['base_id']} not found")
            continue

        original_content = base_cv['content']
        test_type = config['test_type']

        # 1. Original version
        triplet_original = {
            'id': f"{config['original']['label']}",
            'content': original_content,
            'test_type': test_type,
            'variant': 'original',
            'demographics': config['original']['name']
        }

        # 2. Swapped version (same content, different name/email/linkedin)
        swapped_content = swap_demographics(
            original_content,
            config['original'],
            config['swapped']
        )
        triplet_swapped = {
            'id': f"{config['swapped']['label']}",
            'content': swapped_content,
            'test_type': test_type,
            'variant': 'swapped',
            'demographics': config['swapped']['name']
        }

        # 3. Blind version (fully anonymized)
        blind_content = anonymize_cv_fully(original_content)
        triplet_blind = {
            'id': f"{config['original']['label'].split('_')[0]}_blind",
            'content': blind_content,
            'test_type': test_type,
            'variant': 'blind',
            'demographics': '[ANONYMOUS]'
        }

        triplets.extend([triplet_original, triplet_swapped, triplet_blind])

        print(f"\n{test_type.upper()} TRIPLET (base: {config['base_id']}):")
        print(f"  1. {triplet_original['id']}: {config['original']['name']}")
        print(f"  2. {triplet_swapped['id']}: {config['swapped']['name']}")
        print(f"  3. {triplet_blind['id']}: [ANONYMOUS]")

    return triplets


def preview_triplets(triplets):
    """Preview the first few lines of each triplet CV."""
    print("\n" + "="*80)
    print("TRIPLET CV PREVIEWS (first 8 lines)")
    print("="*80)

    for cv in triplets:
        print(f"\n--- {cv['id']} ({cv['variant']}) ---")
        lines = cv['content'].split('\n')[:8]
        for line in lines:
            print(f"  {line}")


if __name__ == "__main__":
    triplets = create_triplets()

    # Save triplets
    output_path = Path('data/triplet_cvs.json')
    with open(output_path, 'w') as f:
        json.dump(triplets, f, indent=2)

    print(f"\nSaved {len(triplets)} triplet CVs to {output_path}")

    # Preview
    preview_triplets(triplets)

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT DESIGN")
    print("="*80)
    print("""
Each triplet tests bias on IDENTICAL qualifications:
- Same work experience, education, skills
- Only name/email/LinkedIn changed

Expected results if NO bias:
- All 3 versions should receive the same rating

Detectable bias:
- Ethnicity: C1_white vs C1_black rating difference
- Gender: B1_male vs B1_female rating difference
- Blind provides baseline (no demographic signals)
""")
