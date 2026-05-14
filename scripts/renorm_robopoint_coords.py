#!/usr/bin/env python3
"""
Renormalize RoboPoint coordinates from 0-1 to 0-1000 for Qwen3-VL.
Save as JSONL file.
"""

import json
import re
import os

BASE = '/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-RoboPoint'

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

src_path = f'{BASE}/robopoint_1432k.json'
dst_path = f'{BASE}/robopoint_1432k_qwen.jsonl'

print(f'Renormalizing RoboPoint coordinates...')
data = json.load(open(src_path, encoding='utf-8'))

with open(dst_path, 'w', encoding='utf-8') as f:
    for entry in data:
        for turn in entry['conversations']:
            # Renormalize coordinates
            turn['value'] = renorm_coords(turn['value'])
            # Update instruction text
            turn['value'] = renorm_instructions(turn['value'])
        # Write as JSONL (one JSON object per line)
        if entry.get('image'):
            entry["images"] = [entry["image"]]
            del entry["image"]
        else:
            continue  # Skip entries without images
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f'✓ Renormalized {len(data)} entries to {dst_path}')

