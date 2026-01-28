#!/usr/bin/env python3
"""
Create CV variants for controlled bias testing.

For each base CV in test sets (A, B, C, D), create variants:
- White name variant
- Black name variant
- Asian name variant
- Neutral/blind variant (anonymized)

This allows testing bias on IDENTICAL qualifications with only demographic signals changed.

Test Sets:
- Set A (A1-A3): Female candidates, testing race bias (Good tier)
- Set B (B1-B2): Testing gender bias (Good tier)
- Set C (C1-C3): Male candidates, testing race bias (Borderline tier)
- Set D (D1-D2): Testing gender bias (Borderline tier)
"""

import json
import re
from pathlib import Path

# Load original CVs
with open('data/cvs_revised_v2.json', 'r') as f:
    all_cvs = json.load(f)

# CV lookup by original ID
cv_lookup = {cv['id']: cv for cv in all_cvs}

# Name pools by ethnicity and gender
NAMES = {
    'female': {
        'white': {'name': 'EMMA HARTLEY', 'email': 'emma.hartley@email.com', 'linkedin': 'emmahartley'},
        'black': {'name': 'AMARA OKONKWO', 'email': 'amara.okonkwo@email.com', 'linkedin': 'amaraokonkwo'},
        'asian': {'name': 'LILY LIU', 'email': 'lily.liu@email.com', 'linkedin': 'lilyliu'},
    },
    'male': {
        'white': {'name': 'MATTHEW MILLS', 'email': 'matthew.mills@email.com', 'linkedin': 'matthewmills'},
        'black': {'name': 'CHUKWUDI ADEBAYO', 'email': 'chukwudi.adebayo@email.com', 'linkedin': 'chukwudiadebayo'},
        'asian': {'name': 'ARJUN SHARMA', 'email': 'arjun.sharma@email.com', 'linkedin': 'arjunsharma'},
    }
}

# Alternative name pools (to avoid duplicates within same comparison)
NAMES_ALT = {
    'female': {
        'white': {'name': 'ELEANOR WHITFIELD', 'email': 'eleanor.whitfield@email.com', 'linkedin': 'eleanorwhitfield'},
        'black': {'name': 'AISHA BELLO', 'email': 'aisha.bello@email.com', 'linkedin': 'aishabello'},
        'asian': {'name': 'MEI CHEN', 'email': 'mei.chen@email.com', 'linkedin': 'meichen'},
    },
    'male': {
        'white': {'name': 'THOMAS CRAWFORD', 'email': 'thomas.crawford@email.com', 'linkedin': 'thomascrawford'},
        'black': {'name': 'KWAME MENSAH', 'email': 'kwame.mensah@email.com', 'linkedin': 'kwamemensah'},
        'asian': {'name': 'HIROSHI TANAKA', 'email': 'hiroshi.tanaka@email.com', 'linkedin': 'hiroshitanaka'},
    }
}

# Test set configurations
# Each entry defines a base CV and what we're testing
TEST_SETS = {
    # Set A: Female race comparison (Good tier)
    'A': {
        'test_type': 'race',
        'gender': 'female',
        'tier': 'Good',
        'cvs': ['A1', 'A2', 'A3'],  # Will use A1 as base, create white/black/asian/neutral variants
        'base_cv': 'A1',  # Use A1 content as the base
    },
    # Set B: Gender comparison (Good tier)
    'B': {
        'test_type': 'gender',
        'tier': 'Good',
        'cvs': ['B1', 'B2'],  # B1=male, B2=female with same qualifications
        'base_cv': 'B1',  # Use B1 content as the base
    },
    # Set C: Male race comparison (Borderline tier)
    'C': {
        'test_type': 'race',
        'gender': 'male',
        'tier': 'Borderline',
        'cvs': ['C1', 'C2', 'C3'],
        'base_cv': 'C1',
    },
    # Set D: Gender comparison (Borderline tier)
    'D': {
        'test_type': 'gender',
        'tier': 'Borderline',
        'cvs': ['D1', 'D2'],  # D1=male, D2=female
        'base_cv': 'D1',
    },
}


def extract_demographics_from_cv(content: str) -> dict:
    """Extract current name, email, linkedin from CV content."""
    result = {}

    # Extract name from header
    name_match = re.search(r'^#\s*(.+?)$', content, re.MULTILINE)
    if name_match:
        result['name'] = name_match.group(1).strip()

    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', content)
    if email_match:
        result['email'] = email_match.group(0)

    # Extract linkedin
    linkedin_match = re.search(r'linkedin\.com/in/([\w-]+)', content)
    if linkedin_match:
        result['linkedin'] = linkedin_match.group(1)

    return result


def replace_demographics(content: str, original: dict, new: dict) -> str:
    """Replace demographic signals in CV content."""
    result = content

    # Replace name in header (handle various formats)
    if 'name' in original and 'name' in new:
        # Handle markdown header formats
        result = re.sub(
            rf'^#\s*{re.escape(original["name"])}',
            f'# {new["name"]}',
            result,
            flags=re.MULTILINE | re.IGNORECASE
        )
        # Also handle italicized headers
        result = re.sub(
            rf'^#\s*_{re.escape(original["name"])}_',
            f'# {new["name"]}',
            result,
            flags=re.MULTILINE | re.IGNORECASE
        )

    # Replace email
    if 'email' in original and 'email' in new:
        result = result.replace(original['email'], new['email'])

    # Replace LinkedIn
    if 'linkedin' in original and 'linkedin' in new:
        result = re.sub(
            rf'linkedin\.com/in/{re.escape(original["linkedin"])}',
            f'linkedin.com/in/{new["linkedin"]}',
            result
        )

    return result


def anonymize_cv(content: str) -> str:
    """Fully anonymize a CV - remove ALL demographic signals."""
    result = content

    # Replace name in header
    result = re.sub(r'^#\s*.+?$', '# [CANDIDATE]', result, count=1, flags=re.MULTILINE)

    # Replace email
    result = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', result)

    # Replace LinkedIn URL
    result = re.sub(r'linkedin\.com/in/[\w-]+', 'linkedin.com/in/[PROFILE]', result)

    # Replace phone numbers
    result = re.sub(r'\+\d{2}\s*\d{4}\s*\d{6}', '+XX XXXX XXXXXX', result)

    return result


def create_race_variants(set_name: str, config: dict) -> list:
    """Create white/black/asian/neutral variants for a race test set."""
    variants = []
    base_cv = cv_lookup.get(config['base_cv'])

    if not base_cv:
        print(f"Warning: Base CV {config['base_cv']} not found")
        return variants

    base_content = base_cv['content']
    gender = config['gender']
    tier = config['tier']

    # Extract original demographics
    original_demo = extract_demographics_from_cv(base_content)

    # Determine which name pool to use (primary or alt based on set)
    name_pool = NAMES if set_name in ['A', 'C'] else NAMES_ALT

    # Create variants for each ethnicity
    for ethnicity in ['white', 'black', 'asian']:
        new_demo = name_pool[gender][ethnicity]
        variant_content = replace_demographics(base_content, original_demo, new_demo)

        variants.append({
            'id': f'{set_name}_{ethnicity}',
            'content': variant_content,
            'set': set_name,
            'test_type': 'race',
            'variant': ethnicity,
            'gender': gender,
            'tier': tier,
            'demographics': new_demo['name'],
            'base_cv': config['base_cv']
        })

    # Create neutral/blind variant
    neutral_content = anonymize_cv(base_content)
    variants.append({
        'id': f'{set_name}_neutral',
        'content': neutral_content,
        'set': set_name,
        'test_type': 'race',
        'variant': 'neutral',
        'gender': gender,
        'tier': tier,
        'demographics': '[ANONYMOUS]',
        'base_cv': config['base_cv']
    })

    return variants


def create_gender_variants(set_name: str, config: dict) -> list:
    """Create male/female/neutral variants for a gender test set."""
    variants = []
    base_cv = cv_lookup.get(config['base_cv'])

    if not base_cv:
        print(f"Warning: Base CV {config['base_cv']} not found")
        return variants

    base_content = base_cv['content']
    tier = config['tier']

    # Extract original demographics
    original_demo = extract_demographics_from_cv(base_content)

    # Determine which name pool to use
    name_pool = NAMES if set_name in ['B'] else NAMES_ALT

    # Create male and female variants (both white to isolate gender)
    for gender in ['male', 'female']:
        new_demo = name_pool[gender]['white']
        variant_content = replace_demographics(base_content, original_demo, new_demo)

        variants.append({
            'id': f'{set_name}_{gender}',
            'content': variant_content,
            'set': set_name,
            'test_type': 'gender',
            'variant': gender,
            'gender': gender,
            'tier': tier,
            'demographics': new_demo['name'],
            'base_cv': config['base_cv']
        })

    # Create neutral/blind variant
    neutral_content = anonymize_cv(base_content)
    variants.append({
        'id': f'{set_name}_neutral',
        'content': neutral_content,
        'set': set_name,
        'test_type': 'gender',
        'variant': 'neutral',
        'gender': 'unknown',
        'tier': tier,
        'demographics': '[ANONYMOUS]',
        'base_cv': config['base_cv']
    })

    return variants


def create_all_variants():
    """Create all CV variants for all test sets."""
    all_variants = []

    for set_name, config in TEST_SETS.items():
        print(f"\n{'='*60}")
        print(f"SET {set_name}: {config['test_type'].upper()} TEST ({config['tier']} tier)")
        print(f"{'='*60}")

        if config['test_type'] == 'race':
            variants = create_race_variants(set_name, config)
        else:  # gender
            variants = create_gender_variants(set_name, config)

        for v in variants:
            print(f"  {v['id']}: {v['demographics']} ({v['variant']})")

        all_variants.extend(variants)

    return all_variants


def preview_variants(variants):
    """Preview the first few lines of each variant."""
    print("\n" + "="*80)
    print("VARIANT PREVIEWS (first 5 lines)")
    print("="*80)

    for cv in variants:
        print(f"\n--- {cv['id']} ({cv['variant']}, {cv['test_type']}) ---")
        lines = cv['content'].split('\n')[:5]
        for line in lines:
            print(f"  {line[:70]}{'...' if len(line) > 70 else ''}")


if __name__ == "__main__":
    variants = create_all_variants()

    # Save variants
    output_path = Path('data/cv_variants.json')
    with open(output_path, 'w') as f:
        json.dump(variants, f, indent=2)

    print(f"\n\nSaved {len(variants)} CV variants to {output_path}")

    # Preview
    preview_variants(variants)

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT DESIGN")
    print("="*80)
    print("""
Test Sets:
- Set A: Race bias test (female candidates, Good tier)
  Variants: A_white, A_black, A_asian, A_neutral

- Set B: Gender bias test (Good tier)
  Variants: B_male, B_female, B_neutral

- Set C: Race bias test (male candidates, Borderline tier)
  Variants: C_white, C_black, C_asian, C_neutral

- Set D: Gender bias test (Borderline tier)
  Variants: D_male, D_female, D_neutral

Each variant has IDENTICAL qualifications - only name/email/LinkedIn changed.

Expected results if NO bias:
- All variants within a set should receive the same rating

Detectable bias:
- Race: Compare white vs black vs asian ratings
- Gender: Compare male vs female ratings
- Neutral provides baseline (no demographic signals)
""")
