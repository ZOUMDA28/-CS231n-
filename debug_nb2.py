#!/usr/bin/env python3
"""Check raw JSON of notebook source."""
import json, os

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "notebooks", "part1-deep-learning-basics", "lecture-01-intro", "practice.ipynb"
)

with open(path, 'r', encoding='utf-8') as f:
    raw = f.read()

# Show first code cell's source in raw JSON
nb = json.loads(raw)
cell4 = nb['cells'][4]
print("Cell 4 source type:", type(cell4['source']))
print("Cell 4 source:", repr(cell4['source'])[:200])
