"""
Export SAT parquet dataset:
- Save image_bytes as JPG files
- Generate JSONL with conversation format and image paths
"""
import io
import json
import os
import sys

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

PARQUET = '/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-SAT/SAT_train.parquet'
OUT_DIR = '/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-SAT/exported'
IMG_DIR = os.path.join(OUT_DIR, 'images')
JSONL_PATH = os.path.join(OUT_DIR, 'SAT_train.jsonl')
BATCH_SIZE = 500


def bytes_to_jpg(raw: bytes, path: str):
    img = Image.open(io.BytesIO(raw))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(path, 'JPEG', quality=95)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    pf = pq.ParquetFile(PARQUET)
    total = pf.metadata.num_rows
    print(f"Total rows: {total}")

    written = 0
    skipped = 0
    global_idx = 0

    with open(JSONL_PATH, 'w') as fout, tqdm(total=total, desc='Exporting') as pbar:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            questions = batch.column('question').to_pylist()
            answers_col = batch.column('answers').to_pylist()
            correct_answers = batch.column('correct_answer').to_pylist()
            image_bytes_col = batch.column('image_bytes').to_pylist()

            for i in range(len(batch)):
                idx = global_idx + i
                imgs_raw = image_bytes_col[i]

                if not imgs_raw:
                    skipped += 1
                    pbar.update(1)
                    continue

                img_paths = []
                for img_idx, raw in enumerate(imgs_raw):
                    if raw is None:
                        continue
                    fname = f'{idx:07d}_{img_idx}.jpg'
                    fpath = os.path.join(IMG_DIR, fname)
                    if not os.path.exists(fpath):
                        try:
                            bytes_to_jpg(bytes(raw), fpath)
                        except Exception as e:
                            print(f"  [warn] row {idx} img {img_idx}: {e}", file=sys.stderr)
                            continue
                    img_paths.append(fpath.replace("/cpfs01/cpfs01/datas/molmo2_er_datasets/Molmo2-ER-SAT/", ""))

                if not img_paths:
                    skipped += 1
                    pbar.update(1)
                    continue

                content = questions[i]
                options = answers_col[i]
                if options:
                    content += '\nOptions: ' + ', '.join(options)

                record = {
                    'messages': [
                        {'role': 'user', 'content': content},
                        {'role': 'assistant', 'content': correct_answers[i]},
                    ],
                    'images': img_paths,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                written += 1
                pbar.update(1)

            global_idx += len(batch)

    print(f"\nDone. Written: {written}, Skipped: {skipped}")
    print(f"Images: {IMG_DIR}")
    print(f"JSONL:  {JSONL_PATH}")


if __name__ == '__main__':
    main()
