#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Re-shard parquet files whose largest (nested) leaf column exceeds pyarrow's ~2GB
single-array limit, which otherwise triggers:
    ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs
when the file is read with `pq.read_table` / HF's parquet builder (the default loader path).

Strategy: only files whose biggest leaf column exceeds TRIGGER_BYTES are split. Each is read
via `iter_batches` (read_table itself fails on these), accumulated up to a row target chosen so
every output part's biggest leaf stays under SAFE_BYTES, and written as single-row-group parts.
The original is replaced atomically: parts are staged as `.tmp-reshard-*`, every part is then
re-read with `read_table` to prove it is now single-chunk, and only then is the original removed
and the parts renamed into place. On any failure the original is left untouched.

Usage:
    python scripts/reshard_large_parquet.py --dry-run            # show the plan only
    python scripts/reshard_large_parquet.py                      # re-shard all subsets
    python scripts/reshard_large_parquet.py --subsets it2vm vpm  # limit to some subsets
"""
import argparse
import glob
import math
import os

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = '/cpfs01/cpfs01/datas/vlm_dataset/pretrain_vlm_action_mask_tokens_hf'
ALL_SUBSETS = ['action', 'gpm', 'idm', 'it2vm', 'sp', 'vpm']

# pyarrow's int32 offset limit on a single Array's values buffer is 2^31-1 (~2.14GB): a file
# whose biggest leaf exceeds it can't be read single-chunk and trips the nested-chunked error.
# Trigger only on files past that hard limit (leave near-limit-but-working files untouched);
# target parts comfortably under SAFE.
TRIGGER_BYTES = 2_147_483_647
SAFE_BYTES = 1_500_000_000


def leaf_sizes(md):
    """Return (max_leaf_uncompressed_bytes, num_rows) for a parquet FileMetaData."""
    n = md.num_columns
    tot = [0] * n
    for rg in range(md.num_row_groups):
        for i in range(n):
            tot[i] += md.row_group(rg).column(i).total_uncompressed_size
    return (max(tot) if tot else 0), md.num_rows


def reshard_file(fp, dry_run):
    pf = pq.ParquetFile(fp)
    md = pf.metadata
    big, rows = leaf_sizes(md)
    name = os.path.basename(fp)
    if big <= TRIGGER_BYTES or rows <= 1:
        print(f'    skip  {name}: biggest_leaf={big / 1e9:.2f}GB rows={rows} (under trigger)')
        return 0

    nparts = max(2, math.ceil(big / SAFE_BYTES))
    target_rows = math.ceil(rows / nparts)
    compression = (md.row_group(0).column(0).compression or 'SNAPPY').lower()
    print(f'    SPLIT {name}: biggest_leaf={big / 1e9:.2f}GB rows={rows} '
          f'-> {nparts} parts (~{target_rows} rows each, compression={compression})')
    if dry_run:
        return nparts

    schema = pf.schema_arrow
    stem = fp[:-len('.parquet')]
    staged = []  # (tmp_path, final_path)
    buf, buf_rows, part = [], 0, 0

    def flush():
        nonlocal buf, buf_rows, part
        if not buf:
            return
        tmp = f'{stem}.tmp-reshard-r{part:02d}.parquet'
        final = f'{stem}-r{part:02d}.parquet'
        tbl = pa.Table.from_batches(buf, schema=schema)
        pq.write_table(tbl, tmp, row_group_size=tbl.num_rows + 1, compression=compression)
        print(f'      staged {os.path.basename(tmp)} rows={tbl.num_rows}')
        staged.append((tmp, final))
        buf, buf_rows = [], 0
        part += 1

    try:
        for b in pf.iter_batches(batch_size=2000):
            buf.append(b)
            buf_rows += b.num_rows
            if buf_rows >= target_rows:
                flush()
        flush()

        # verify every staged part is now readable single-chunk via read_table
        for tmp, _ in staged:
            pq.read_table(tmp)
        print(f'      verified {len(staged)} parts read single-chunk')
    except Exception:
        for tmp, _ in staged:
            if os.path.exists(tmp):
                os.remove(tmp)
        print(f'      ERROR re-sharding {name}; original left untouched, staged files removed')
        raise

    # atomic-ish swap: drop original, then rename parts into place
    os.remove(fp)
    for tmp, final in staged:
        os.rename(tmp, final)
    print(f'      replaced {name} with {len(staged)} parts')
    return len(staged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--subsets', nargs='*', default=ALL_SUBSETS)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    total_split = 0
    for s in args.subsets:
        files = sorted(glob.glob(os.path.join(args.root, s, '*.parquet')))
        # skip files already produced by a previous run
        files = [f for f in files if '.tmp-reshard-' not in os.path.basename(f)]
        print(f'=== {s} ({len(files)} files) ===')
        for fp in files:
            total_split += 1 if reshard_file(fp, args.dry_run) else 0
    verb = 'would re-shard' if args.dry_run else 're-sharded'
    print(f'\nDone: {verb} {total_split} file(s).')


if __name__ == '__main__':
    main()
