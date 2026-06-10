# Copyright (c) ModelScope Contributors. All rights reserved.
import glob
import os

from datasets import Dataset as HfDataset
from modelscope.hub.utils.utils import get_cache_dir

from swift.dataset.dataset_meta import DatasetMeta
from swift.dataset.loader import DatasetLoader
from swift.dataset.preprocessor import RowPreprocessor
from swift.dataset.register import register_dataset

# VLM/VLA pretrain dataset. Rows already use the standard {messages, images} schema (with
# inline <image>/<video> tags and ROOT_IMAGE_DIR-relative media paths) plus a compact robot
# state payload (`state_b64_f16` + `state_mask_bits`) that is decoded lazily at encode time.
# Multiple HF parquet configs, no `default` -> load a subset explicitly, e.g.
#   --dataset /cpfs01/cpfs01/datas/vlm_dataset/pretrain_vlm_action_mask_tokens_hf:action
# or all subsets with `:all`.
PRETRAIN_VLM_ACTION_DIR = '/cpfs01/cpfs01/datas/vlm_dataset/pretrain_vlm_action_mask_tokens_hf'
PRETRAIN_VLM_ACTION_SUBSETS = ['action', 'gpm', 'idm', 'it2vm', 'sp', 'vpm']


class VLAParquetLoader(DatasetLoader):
    """Read the per-subset parquet shards directly via `pyarrow.iter_batches` + `to_pylist`.

    Some shards (e.g. it2vm, vpm) store the nested `messages`/`images` columns across multiple
    chunks, and HF's parquet builder then trips pyarrow's "Nested data conversions not
    implemented for chunked array outputs". Materializing through python row dicts sidesteps
    that cast entirely (same approach as the molmo2_er parquet loaders).
    """

    def _load_repo_dataset(self, dataset_id, subset, *, use_hf=None, revision=None):
        datasets = []
        for split in subset.split:
            files = sorted(glob.glob(os.path.join(dataset_id, subset.subset, f'{split}-*.parquet')))
            if not files:
                files = sorted(glob.glob(os.path.join(dataset_id, subset.subset, '*.parquet')))
            assert files, f'No parquet files found for subset={subset.subset}, split={split} under {dataset_id}'

            def gen(files):
                import pyarrow.parquet as pq
                for fp in files:
                    for batch in pq.ParquetFile(fp).iter_batches(batch_size=1000):
                        yield from batch.to_pylist()

            num_proc = min(self.num_proc, len(files)) if self.num_proc else None
            dataset = HfDataset.from_generator(
                gen,
                gen_kwargs={'files': files},
                num_proc=num_proc,
                cache_dir=os.path.join(get_cache_dir(), 'datasets'))
            if self.columns:
                dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
            dataset = subset.preprocess_func(
                dataset,
                num_proc=self.num_proc,
                load_from_cache_file=self.load_from_cache_file,
                strict=self.strict,
                enable_auto_mapping=not self.disable_auto_column_mapping)
            if self.remove_unused_columns:
                dataset = RowPreprocessor.remove_useless_columns(dataset)
            datasets.append(dataset)
        return self.concat_datasets(datasets)


register_dataset(
    DatasetMeta(
        dataset_name='pretrain-vlm-action',
        dataset_path=PRETRAIN_VLM_ACTION_DIR,
        subsets=PRETRAIN_VLM_ACTION_SUBSETS,
        loader=DatasetLoader,
        tags=['vla', 'robotics', 'multimodal'],
    ))
