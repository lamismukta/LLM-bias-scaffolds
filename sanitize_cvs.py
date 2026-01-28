#!/usr/bin/env python3
"""Sanitize CV data by randomizing IDs and order."""
import json
import random
import string
from pathlib import Path


def generate_random_id(length=8):
    """Generate a random alphanumeric ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def sanitize_cvs(input_path: str, output_path: str, mapping_path: str):
    """Sanitize CVs by randomizing IDs and order."""
    # Load original CVs
    with open(input_path, 'r') as f:
        original_cvs = json.load(f)
    
    # Create mapping and sanitized CVs
    id_mapping = {}
    sanitized_cvs = []
    
    # Shuffle the order
    shuffled_cvs = original_cvs.copy()
    random.shuffle(shuffled_cvs)
    
    for cv in shuffled_cvs:
        original_id = cv['id']
        new_id = generate_random_id()
        
        # Store mapping
        id_mapping[new_id] = {
            "original_id": original_id,
            "original_name": cv.get('name', 'Unknown')
        }
        
        # Create sanitized CV (only ID and content)
        sanitized_cv = {
            "id": new_id,
            "content": cv['content']
        }
        sanitized_cvs.append(sanitized_cv)
    
    # Save sanitized CVs
    with open(output_path, 'w') as f:
        json.dump(sanitized_cvs, f, indent=2)
    
    # Save mapping
    with open(mapping_path, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    
    print(f"Sanitized {len(sanitized_cvs)} CVs")
    print(f"Sanitized CVs saved to: {output_path}")
    print(f"ID mapping saved to: {mapping_path}")
    print(f"\nExample mapping (first 5):")
    for i, (new_id, info) in enumerate(list(id_mapping.items())[:5]):
        print(f"  {new_id} -> {info['original_id']} ({info['original_name']})")


if __name__ == "__main__":
    input_path = "data/cvs_revised_v2.json"
    output_path = "data/cvs_sanitized.json"
    mapping_path = "data/cv_id_mapping.json"
    
    sanitize_cvs(input_path, output_path, mapping_path)

