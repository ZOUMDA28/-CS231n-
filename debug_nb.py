#!/usr/bin/env python3
"""Debug notebook structure."""
import json
import os

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "notebooks", "part1-deep-learning-basics", "lecture-01-intro", "practice.ipynb"
)

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = cell.get('source', '')
    if isinstance(src, list):
        src = ''.join(src)
    print(f"Cell {i}: type={cell['cell_type']}, source_len={len(src)}")
    if cell['cell_type'] == 'code':
        print(f"  first 80 chars: {repr(src[:80])}")
        print(f"  outputs: {cell.get('outputs', 'NONE')}")
