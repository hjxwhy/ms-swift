#!/usr/bin/env python3
"""
Merge all RefSpatial JSON files into a single JSONL file.
Store image paths as 'images' field with relative paths (image_dir/filename).
"""

import json
import os

BASE = '/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-RefSpatial'

# Map each JSON file to its image directory
FILES = [
    ('2D/reasoning_template_qa_qwen.json', '2D/image'),
    ('2D/choice_qa_qwen.json', '2D/image'),
    ('3D/reasoning_template_qa_qwen.json', '3D/image'),
    ('3D/choice_qa_qwen.json', '3D/image'),
    ('3D/vacant_qa_qwen.json', '3D/image'),
    ('3D/multi_view_qa_qwen.json', '3D/image_multi_view'),
    ('3D/visual_choice_qa_qwen.json', '3D/image_visual_choice'),
]

output_path = f'{BASE}/refspatial_merged.jsonl'

print(f'Merging {len(FILES)} RefSpatial JSON files...')
total_records = 0

with open(output_path, 'w', encoding='utf-8') as out:
    for json_file, image_dir in FILES:
        src_path = os.path.join(BASE, json_file)
        print(f'  Processing {json_file} → {image_dir}')

        data = json.load(open(src_path, encoding='utf-8'))

        for record in data:
            # Store images with relative paths (image_dir/filename)
            clean_record = {
                'id': record.get('id'),
                'conversations': record.get('conversations'),
                'images': [f'{image_dir}/{img}' for img in record.get('image', [])],
            }
            # Write as JSONL (one JSON object per line)
            out.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
            total_records += 1

print(f'\n✓ Merged {total_records} records to {output_path}')






