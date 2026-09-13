#!/usr/bin/env python3
"""Fix lecture-07-training-nn notebook which has malformed code cells."""
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(BASE, "notebooks", "part2-cnn-vision", "lecture-07-training-nn", "practice.ipynb")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def fix_code_source(source):
    """Fix code source by inserting newlines at proper locations."""
    if isinstance(source, list):
        source = ''.join(source)
    
    # If already has reasonable newlines, check for specific issues
    # Common issues from bad generation:
    # 1. "import matplotlibmatplotlib.use('Agg')" -> two statements merged
    # 2. class/def with wrong indentation
    
    lines = source.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix "import matplotlibmatplotlib.use('Agg')"
        line = re.sub(r"import matplotlibmatplotlib\.use\(", "import matplotlib\nmatplotlib.use(", line)
        
        # Fix "import matplotlib.pyplot as pltnp.random.seed"
        line = re.sub(r"import matplotlib\.pyplot as pltnp\.random\.seed", "import matplotlib.pyplot as plt\nnp.random.seed", line)
        
        # Fix "def xxx(...):    # comment" - missing newline after comment
        # (not needed if already on separate lines)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


fixed_count = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    fixed = fix_code_source(source)
    
    if fixed != source:
        cell['source'] = fixed.split('\n')
        # Add newlines
        cell['source'] = [l + '\n' for l in cell['source']]
        fixed_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Fixed {fixed_count} code cells")
