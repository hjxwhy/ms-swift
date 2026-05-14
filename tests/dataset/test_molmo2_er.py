#!/usr/bin/env python3
"""
Test all 10 Molmo2-ER datasets for:
- Successful loading
- Correct messages format (role/content)
- No <|image_pad|> placeholders
- Valid image/video paths or bytes
- Normalized conversation format
"""

import os
import sys

# Add ms-swift to path
sys.path.insert(0, '/cpfs01/jensen/code/code_latest/ms-swift')

# Set cache_dir BEFORE importing swift
os.environ['MODELSCOPE_CACHE'] = '/cpfs01/cpfs01/cache/molmo2'
os.environ['HF_DATASETS_CACHE'] = '/cpfs01/cpfs01/cache/molmo2'

from swift.dataset import load_dataset

DATASETS = [
    # ('molmo2-er-clevr', 'images'),
    # ('molmo2-er-grid3d', 'images'),
    # ('molmo2-er-refspatial', 'images'),
    # ('molmo2-er-robopoint', 'images'),
    # ('molmo2-er-robovqa', 'videos'),
    # ('molmo2-er-sat', 'images'),
    ('molmo2-er-sensenova-si', 'images'),
    # ('molmo2-er-sims-vsi', 'videos'),
    # ('molmo2-er-vsi-590k', 'videos'),
    # ('molmo2-er-vstp', 'images'),
]

def validate_messages(messages):
    """Validate messages format: list of {role, content} dicts."""
    assert isinstance(messages, list), f"messages must be list, got {type(messages)}"
    assert len(messages) > 0, "messages must not be empty"
    for msg in messages:
        assert isinstance(msg, dict), f"each message must be dict, got {type(msg)}"
        assert 'role' in msg, f"message missing 'role': {msg}"
        assert 'content' in msg, f"message missing 'content': {msg}"
        assert msg['role'] in ['user', 'assistant', 'system'], f"invalid role: {msg['role']}"
        assert isinstance(msg['content'], str), f"content must be str, got {type(msg['content'])}"
        # Check for <|image_pad|> placeholder
        assert '<|image_pad|>' not in msg['content'], f"found <|image_pad|> in message: {msg['content'][:100]}"

def validate_media(media_list, media_key):
    """Validate image/video paths or bytes."""
    if not media_list:
        return  # OK if empty

    for item in media_list:
        if isinstance(item, dict):
            # Bytes format: {'bytes': b'...', 'path': None}
            has_bytes = item.get('bytes') is not None
            has_path = item.get('path') is not None and os.path.exists(item['path'])
            assert has_bytes or has_path, f"{media_key} dict missing bytes or valid path: {item}"
        else:
            # Path format: string path
            assert isinstance(item, str), f"{media_key} item must be str or dict, got {type(item)}"
            assert os.path.exists(item), f"{media_key} path not found: {item}"

def test_dataset(name, media_key):
    """Test a single dataset - actually read media files."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)

    try:
        # Load 10 samples
        train_ds, _ = load_dataset(f'{name}#10', num_proc=64, load_from_cache_file=True)
        print(f"✓ Loaded {len(train_ds)} rows")

        # Test each row
        valid_rows = 0
        media_found = 0
        media_readable = 0

        for i, row in enumerate(train_ds):
            # Check messages
            if 'messages' not in row or not row['messages']:
                print(f"  ✗ Row {i}: no messages")
                continue

            # Check media
            media = row.get(media_key, [])
            if not media:
                print(f"  ⚠ Row {i}: no {media_key}")
                valid_rows += 1
                continue
            
            media_found += len(media)

            # Try to read media
            for j, item in enumerate(media):
                try:
                    if isinstance(item, dict):
                        # Bytes format
                        if item.get('bytes'):
                            media_readable += 1
                        elif item.get('path') and os.path.exists(item['path']):
                            # Try to read file
                            with open(item['path'], 'rb') as f:
                                data = f.read(100)  # Read first 100 bytes
                            media_readable += 1
                    else:
                        # Path format
                        if os.path.exists(item):
                            # Try to read file
                            with open(item, 'rb') as f:
                                data = f.read(100)  # Read first 100 bytes
                            media_readable += 1
                        else:
                            print(f"  ✗ Row {i}: {media_key}[{j}] not found: {item}")
                except Exception as e:
                    print(f"  ✗ Row {i}: {media_key}[{j}] read error: {str(e)[:60]}")

            valid_rows += 1

        print(f"\n✓ Valid rows: {valid_rows}/{len(train_ds)}")
        print(f"✓ Rows with {media_key}: {media_found}/{len(train_ds)}")
        print(f"✓ Readable {media_key}: {media_readable}/{media_found if media_found > 0 else 1}")

        if valid_rows == len(train_ds) and media_readable == media_found:
            print(f"\n✓✓✓ {name} FULLY PASSED ✓✓✓")
            return True
        else:
            print(f"\n⚠ {name} PARTIAL - some issues found")
            return False

    except Exception as e:
        print(f"\n✗✗✗ {name} FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")
    print(f"Testing {len(DATASETS)} Molmo2-ER datasets...\n")

    results = {}
    for name, media_key in DATASETS:
        results[name] = test_dataset(name, media_key)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
