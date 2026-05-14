#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path


DEFAULT_WITH_OPTIONS = (
    '/cpfs01/cpfs01/datas/vlm_dataset/subtask_relabel/subtask_dataset_with_options.json'
)
DEFAULT_WITHOUT_OPTIONS = (
    '/cpfs01/cpfs01/datas/vlm_dataset/subtask_relabel/subtask_dataset_without_options.json'
)
DEFAULT_OUTPUT = 'output/subtask_dataset_30with_70without_unique.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample a unique mixed subtask dataset by videos path.')
    parser.add_argument('--with-options', default=DEFAULT_WITH_OPTIONS)
    parser.add_argument('--without-options', default=DEFAULT_WITHOUT_OPTIONS)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--with-ratio',
        type=float,
        default=0.3,
        help='Ratio sampled from the with-options dataset in the final output.')
    parser.add_argument(
        '--total-size',
        type=int,
        default=None,
        help='Final sample size. Defaults to the number of unique videos across both inputs.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation. Use 0 for compact JSON.')
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f'{path} must contain a top-level JSON list.')
    return data


def video_key(item):
    videos = item.get('videos')
    if videos is None:
        raise KeyError(f'Item missing "videos": {item}')
    if isinstance(videos, list):
        return '\n'.join(str(video) for video in videos)
    return str(videos)


def build_video_index(data, name):
    index = OrderedDict()
    duplicates = []
    for item in data:
        key = video_key(item)
        if key in index:
            duplicates.append(key)
            continue
        index[key] = item
    if duplicates:
        raise ValueError(f'{name} contains duplicate videos paths, first duplicate: {duplicates[0]}')
    return index


def split_counts(total_size, with_ratio):
    if not 0 <= with_ratio <= 1:
        raise ValueError('--with-ratio must be in [0, 1].')
    with_count = round(total_size * with_ratio)
    return with_count, total_size - with_count


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    with_data = load_json(args.with_options)
    without_data = load_json(args.without_options)
    with_index = build_video_index(with_data, 'with-options dataset')
    without_index = build_video_index(without_data, 'without-options dataset')

    all_keys = list(OrderedDict.fromkeys([*with_index.keys(), *without_index.keys()]))
    total_size = args.total_size if args.total_size is not None else len(all_keys)
    if total_size < 0:
        raise ValueError('--total-size must be >= 0.')
    if total_size > len(all_keys):
        raise ValueError(
            f'--total-size={total_size} exceeds unique videos count {len(all_keys)}.')

    with_count, without_count = split_counts(total_size, args.with_ratio)
    rng.shuffle(all_keys)

    selected_with_keys = []
    selected_without_keys = []
    used_keys = set()

    for key in all_keys:
        if len(selected_with_keys) >= with_count:
            break
        if key in with_index:
            selected_with_keys.append(key)
            used_keys.add(key)

    for key in all_keys:
        if len(selected_without_keys) >= without_count:
            break
        if key in used_keys:
            continue
        if key in without_index:
            selected_without_keys.append(key)
            used_keys.add(key)

    if len(selected_with_keys) < with_count or len(selected_without_keys) < without_count:
        raise RuntimeError(
            'Unable to satisfy the requested ratio without duplicate videos paths: '
            f'need {with_count} with-options and {without_count} without-options, got '
            f'{len(selected_with_keys)} and {len(selected_without_keys)}.')

    mixed_data = [with_index[key] for key in selected_with_keys]
    mixed_data.extend(without_index[key] for key in selected_without_keys)
    rng.shuffle(mixed_data)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent == 0 else args.indent
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(mixed_data, f, ensure_ascii=False, indent=indent)
        f.write('\n')

    print(f'with_options rows: {len(with_data)}, unique videos: {len(with_index)}')
    print(f'without_options rows: {len(without_data)}, unique videos: {len(without_index)}')
    print(f'output rows: {len(mixed_data)}')
    print(f'with_options sampled: {len(selected_with_keys)}')
    print(f'without_options sampled: {len(selected_without_keys)}')
    print(f'unique output videos: {len({video_key(item) for item in mixed_data})}')
    print(f'output: {output}')


if __name__ == '__main__':
    main()
