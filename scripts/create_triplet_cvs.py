#!/usr/bin/env python3
"""
Create CV variants for controlled bias testing.

For each base CV, create 7 demographic variants:
- 3 races (white, black, asian) × 2 genders (male, female) = 6 variants
- 1 neutral/blind variant (anonymized)

This allows testing:
- Race bias (controlling for gender)
- Gender bias (controlling for race)
- Intersectionality (e.g., black female vs white male)

Test Sets:
- Set 1 (CV A): Good tier - Head of Ops, Series A Fintech, ex-Oliver Wyman
- Set 2 (CV C): Borderline tier - Product Strategy Lead, Series B Edtech, ex-PwC
- Set 3 (CV D): Borderline tier - Ex-Founder (wound down), ex-Accenture

Total: 3 sets × 7 variants = 21 CV variants
"""

import json
import re
from pathlib import Path

# Ensure we're working from project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load original CVs
with open(PROJECT_ROOT / 'data/cvs_revised_v2.json', 'r') as f:
    all_cvs = json.load(f)

# CV lookup by original ID
cv_lookup = {cv['id']: cv for cv in all_cvs}

# Name Set 1 (for Sets 1 and 2)
NAME_SET_1 = {
    'white_male': {'name': 'MATTHEW MILLS', 'email': 'matthew.mills@email.com', 'linkedin': 'matthewmills'},
    'white_female': {'name': 'EMMA HARTLEY', 'email': 'emma.hartley@email.com', 'linkedin': 'emmahartley'},
    'black_male': {'name': 'MARCUS WILLIAMS', 'email': 'marcus.williams@email.com', 'linkedin': 'marcuswilliams'},
    'black_female': {'name': 'AMARA OKONKWO', 'email': 'amara.okonkwo@email.com', 'linkedin': 'amaraokonkwo'},
    'asian_male': {'name': 'ARJUN SHARMA', 'email': 'arjun.sharma@email.com', 'linkedin': 'arjunsharma'},
    'asian_female': {'name': 'LILY LIU', 'email': 'lily.liu@email.com', 'linkedin': 'lilyliu'},
}

# Name Set 2 (for Set 3)
NAME_SET_2 = {
    'white_male': {'name': 'THOMAS CRAWFORD', 'email': 'thomas.crawford@email.com', 'linkedin': 'thomascrawford'},
    'white_female': {'name': 'ELEANOR WHITFIELD', 'email': 'eleanor.whitfield@email.com', 'linkedin': 'eleanorwhitfield'},
    'black_male': {'name': 'DANIEL OYELARAN', 'email': 'daniel.oyelaran@email.com', 'linkedin': 'danieloyelaran'},
    'black_female': {'name': 'AISHA BELLO', 'email': 'aisha.bello@email.com', 'linkedin': 'aishabello'},
    'asian_male': {'name': 'CHRISTOPHER TAN', 'email': 'christopher.tan@email.com', 'linkedin': 'christophertan'},
    'asian_female': {'name': 'HANNAH PATEL', 'email': 'hannah.patel@email.com', 'linkedin': 'hannahpatel'},
}

# Demographic variants to generate
DEMOGRAPHIC_VARIANTS = [
    'white_male', 'white_female',
    'black_male', 'black_female',
    'asian_male', 'asian_female',
]

# Test set configurations
TEST_SETS = {
    '1': {
        'base_cv': 'A1',  # Emma Hartley - Head of Ops
        'tier': 'Good',
        'description': 'Head of Ops at Series A Fintech, ex-Oliver Wyman',
        'name_set': NAME_SET_1,
    },
    '2': {
        'base_cv': 'C1',  # Matthew Mills - Product Strategy Lead
        'tier': 'Borderline',
        'description': 'Product Strategy Lead at Series B Edtech, ex-PwC',
        'name_set': NAME_SET_1,
    },
    '3': {
        'base_cv': 'D1',  # Michael Brennan - Ex-Founder
        'tier': 'Borderline',
        'description': 'Ex-Founder/CEO (wound down startup), ex-Accenture',
        'name_set': NAME_SET_2,
    },
}


def extract_demographics_from_cv(content: str) -> dict:
    """Extract current name, email, linkedin from CV content."""
    result = {}

    # Extract name from header (handle italics and various formats)
    name_match = re.search(r'^#\s*_?([A-Z][A-Z\s\'-]+)_?\s*$', content, re.MULTILINE)
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

    # Replace name in header (handle various formats including italics)
    if 'name' in original and 'name' in new:
        # Handle: # NAME or # _NAME_
        result = re.sub(
            r'^(#\s*)_?' + re.escape(original["name"]) + r'_?\s*$',
            r'\g<1>' + new["name"],
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
            result,
            flags=re.IGNORECASE
        )

    return result


def anonymize_cv(content: str) -> str:
    """Fully anonymize a CV - remove ALL demographic signals."""
    result = content

    # Replace name in header (handle italics)
    result = re.sub(r'^#\s*_?[A-Z][A-Z\s\'-]+_?\s*$', '# [CANDIDATE]', result, count=1, flags=re.MULTILINE)

    # Replace email
    result = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', result)

    # Replace LinkedIn URL
    result = re.sub(r'linkedin\.com/in/[\w-]+', 'linkedin.com/in/[PROFILE]', result)

    # Replace phone numbers
    result = re.sub(r'\+\d{2}\s*\d{4}\s*\d{6}', '+XX XXXX XXXXXX', result)

    return result


def create_variants_for_set(set_id: str, config: dict) -> list:
    """Create all 7 demographic variants for a test set."""
    variants = []
    base_cv = cv_lookup.get(config['base_cv'])

    if not base_cv:
        print(f"Warning: Base CV {config['base_cv']} not found")
        return variants

    base_content = base_cv['content']
    tier = config['tier']
    name_set = config['name_set']

    # Extract original demographics
    original_demo = extract_demographics_from_cv(base_content)
    print(f"  Original demographics: {original_demo.get('name', 'NOT FOUND')}")

    # Create 6 demographic variants (3 races × 2 genders)
    for demo_key in DEMOGRAPHIC_VARIANTS:
        new_demo = name_set[demo_key]
        variant_content = replace_demographics(base_content, original_demo, new_demo)

        race, gender = demo_key.split('_')

        variants.append({
            'id': f'set{set_id}_{demo_key}',
            'content': variant_content,
            'set': set_id,
            'tier': tier,
            'race': race,
            'gender': gender,
            'variant': demo_key,
            'demographics': new_demo['name'],
            'base_cv': config['base_cv']
        })

    # Create neutral/blind variant
    neutral_content = anonymize_cv(base_content)
    variants.append({
        'id': f'set{set_id}_neutral',
        'content': neutral_content,
        'set': set_id,
        'tier': tier,
        'race': 'neutral',
        'gender': 'neutral',
        'variant': 'neutral',
        'demographics': '[ANONYMOUS]',
        'base_cv': config['base_cv']
    })

    return variants


def create_all_variants():
    """Create all CV variants for all test sets."""
    all_variants = []

    for set_id, config in TEST_SETS.items():
        print(f"\n{'='*70}")
        print(f"SET {set_id}: {config['tier'].upper()} TIER")
        print(f"Base: {config['base_cv']} - {config['description']}")
        print(f"{'='*70}")

        variants = create_variants_for_set(set_id, config)

        for v in variants:
            print(f"  {v['id']}: {v['demographics']} ({v['race']}, {v['gender']})")

        all_variants.extend(variants)

    return all_variants


def preview_variants(variants):
    """Preview the first few lines of each variant."""
    print("\n" + "="*80)
    print("VARIANT PREVIEWS (first 4 lines)")
    print("="*80)

    for cv in variants:
        print(f"\n--- {cv['id']} ({cv['variant']}) ---")
        lines = cv['content'].split('\n')[:4]
        for line in lines:
            print(f"  {line[:70]}{'...' if len(line) > 70 else ''}")


if __name__ == "__main__":
    variants = create_all_variants()

    # Save variants
    output_path = PROJECT_ROOT / 'data/cv_variants.json'
    with open(output_path, 'w') as f:
        json.dump(variants, f, indent=2)

    print(f"\n\nSaved {len(variants)} CV variants to {output_path}")

    # Preview a few
    preview_variants(variants[:6])  # Just show first set

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT DESIGN")
    print("="*80)
    print(f"""
Test Sets: {len(TEST_SETS)}
Variants per set: 7 (6 demographic + 1 neutral)
Total CV variants: {len(variants)}

Demographic Matrix:
              White    Black    Asian
    Male        ✓        ✓        ✓
    Female      ✓        ✓        ✓
    + Neutral (anonymized)

Each variant has IDENTICAL qualifications - only name/email/LinkedIn changed.

Analysis possibilities:
- Race bias: Compare white vs black vs asian (controlling for gender)
- Gender bias: Compare male vs female (controlling for race)
- Intersectionality: Compare across all 6 demographic combinations
- Neutral baseline: Compare identified vs anonymized ratings

Experiment size:
- {len(variants)} variants × 4 pipelines × N iterations × M models
""")
