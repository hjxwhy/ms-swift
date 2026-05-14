#!/usr/bin/env python3
"""
Clean VSI-590K JSONL - remove extra 'image' field and rename 'video' to 'videos'.
"""

import json

src_path = '/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-VSI-590K/vsi_590k.jsonl'
dst_path = '/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-VSI-590K/vsi_590k_clean.jsonl'

print(f'Cleaning VSI-590K JSONL...')
total = 0

with open(src_path) as f_in, open(dst_path, 'w') as f_out:
    for line in f_in:
        record = json.loads(line)
        # Keep only necessary fields, rename video to videos
        clean_record = {
            'conversations': record.get('conversations'),
            'videos': [record.get('video')],  # Convert to list
            'question_type': record.get('question_type'),
        }
        f_out.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
        total += 1

print(f'✓ Cleaned {total} records to {dst_path}')

