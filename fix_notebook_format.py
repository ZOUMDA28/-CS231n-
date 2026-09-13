#!/usr/bin/env python3
"""Fix notebooks where code cell sources have no newlines (all on one line).
Detects Python keywords and inserts newlines appropriately.
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE, "notebooks")


def fix_code_source(source):
    """Try to fix a code source string that has all code on one line."""
    if isinstance(source, list):
        source = ''.join(source)
    
    # If already has newlines, return as-is
    if '\n' in source:
        return source
    
    # Strategy: insert newlines before Python statement keywords
    # List of patterns that should start a new line
    patterns = [
        # Import statements
        (r'(?<=.)(import\s)', r'\n\1'),
        (r'(?<=.)(from\s+\S+\s+import\s)', r'\n\1'),
        
        # Function/class definitions
        (r'(?<=.)(def\s+\w+\s*\()', r'\n\1'),
        (r'(?<=.)(class\s+\w+)', r'\n\1'),
        
        # Control flow
        (r'(?<=.)(if\s+.*?:)', r'\n\1'),
        (r'(?<=.)(elif\s+.*?:)', r'\n\1'),
        (r'(?<=.)(else\s*:)', r'\n\1'),
        (r'(?<=.)(for\s+.*?:)', r'\n\1'),
        (r'(?<=.)(while\s+.*?:)', r'\n\1'),
        (r'(?<=.)(try\s*:)', r'\n\1'),
        (r'(?<=.)(except\s*.*?:)', r'\n\1'),
        (r'(?<=.)(finally\s*:)', r'\n\1'),
        (r'(?<=.)(with\s+.*?:)', r'\n\1'),
        
        # Return/yield
        (r'(?<=.)(return\s)', r'\n\1'),
        (r'(?<=.)(yield\s)', r'\n\1'),
        
        # Comments
        (r'(?<=.)(#\s)', r'\n\1'),
        
        # Assignment blocks with comments before them
        (r'(?<=.)(#\s*=+)', r'\n\1'),
        
        # Print statements
        (r'(?<=.)(print\()', r'\n\1'),
        
        # np.random.seed
        (r'(?<=.)(np\.random\.seed\()', r'\n\1'),
    ]
    
    result = source
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
    # Clean up: remove leading newline if present
    result = result.lstrip('\n')
    
    # Split into lines
    lines = result.split('\n')
    
    return lines


def fix_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    fixed = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        
        source = cell['source']
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        # Check if it needs fixing (no newlines but has significant content)
        if '\n' not in source_str and len(source_str) > 100:
            fixed_lines = fix_code_source(source_str)
            if isinstance(fixed_lines, list) and len(fixed_lines) > 1:
                # Add newlines back
                cell['source'] = [line + '\n' for line in fixed_lines]
                fixed += 1
    
    if fixed > 0:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"  Fixed {fixed} code cells")
    
    return fixed


def main():
    total_fixed = 0
    for root, dirs, files in os.walk(NOTEBOOKS_DIR):
        for f in files:
            if f.endswith('.ipynb'):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, BASE)
                print(f"Checking: {rel}")
                n = fix_notebook(path)
                if n > 0:
                    total_fixed += n
    
    print(f"\nTotal cells fixed: {total_fixed}")


if __name__ == '__main__':
    main()
