#!/usr/bin/env python3
"""
Renormalize RefSpatial coordinates from 0-1 to 0-1000 for Qwen3-VL.

Converts all 7 RefSpatial JSON QA files:
- Replaces (0.xxx, 0.yyy) with (xxx, yyy) by multiplying by 1000
- Updates instruction text "between 0 and 1" → "between 0 and 1000"
- Saves as *_qwen.json alongside originals (originals untouched)
"""

import json
import re
import os

BASE = '/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-RefSpatial'

# Regex to match numeric float coordinates like (0.502, 0.775)
COORD_PAT = re.compile(r'\((\d+\.\d+),\s*(\d+\.\d+)\)')

def renorm_coords(text):
    """Replace (0.xxx, 0.yyy) with (xxx, yyy) scaled to 0-1000."""
    def replace(m):
        x = round(float(m.group(1)) * 1000)
        y = round(float(m.group(2)) * 1000)
        return f'({x}, {y})'
    return COORD_PAT.sub(replace, text)

def renorm_instructions(text):
    """Update instruction text from 0-1 range to 0-1000 range."""
    text = text.replace('between 0 and 1,', 'between 0 and 1000,')
    text = text.replace('between 0 and 1.', 'between 0 and 1000.')
    return text

# All 7 files (5 with coords, 2 without)
FILES = [
    '2D/reasoning_template_qa.json',
    '2D/choice_qa.json',
    '3D/reasoning_template_qa.json',
    '3D/choice_qa.json',
    '3D/vacant_qa.json',
    '3D/multi_view_qa.json',
    '3D/visual_choice_qa.json',
]

for rel_path in FILES:
    src = os.path.join(BASE, rel_path)
    dst = src.replace('.json', '_qwen.json')

    print(f'Processing {rel_path} ...', flush=True)
    data = json.load(open(src, encoding='utf-8'))

    for entry in data:
        for turn in entry['conversations']:
            # Renormalize coordinates
            turn['value'] = renorm_coords(turn['value'])
            # Update instruction text
            turn['value'] = renorm_instructions(turn['value'])

    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'), indent=2)

    print(f'  ✓ {dst}  ({len(data)} entries)')

print('\nDone! All files converted to *_qwen.json format.')
