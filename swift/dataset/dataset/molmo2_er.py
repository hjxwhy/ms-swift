import glob
import os
from typing import Optional

from datasets import load_dataset as hf_load_dataset
from modelscope.hub.utils.utils import get_cache_dir

from swift.dataset.dataset_meta import DatasetMeta
from swift.dataset.loader import DatasetLoader
from swift.dataset.preprocessor import RowPreprocessor
from swift.dataset.register import register_dataset

BASE = 'molmo2_er_datasets'


def _conv_to_messages(conversations):
    role_map = {'human': 'user', 'gpt': 'assistant', 'system': 'system',
                'user': 'user', 'assistant': 'assistant'}
    return [{'role': role_map[c.get('from') or c['role']],
             'content': c['value'] if 'value' in c else c['content']}
            for c in conversations]


class CLEVRPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        img = f"{BASE}/Molmo2-ER-CLEVR/CLEVR_v1.0/images/train/{row['image_filename']}"
        return {
            'messages': [{'role': 'user', 'content': '<image>' + row['question']},
                         {'role': 'assistant', 'content': str(row['answer'])}],
            'images': [img],
        }


class CLEVRLoader(DatasetLoader):
    def _load_dataset_path(self, dataset_path, dataset_meta):
        from swift.utils import safe_ddp_context
        kwargs = {'split': 'train', 'streaming': self.streaming, 'num_proc': self.num_proc,
                  'field': 'questions'}
        with safe_ddp_context(None, True):
            kwargs['cache_dir'] = os.path.join(get_cache_dir(), 'datasets')
            dataset = hf_load_dataset('json', data_files=dataset_path, **kwargs)
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file,
            strict=self.strict, enable_auto_mapping=not self.disable_auto_column_mapping)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset


class GRiD3DPreprocessor(RowPreprocessor):
    def prepare_dataset(self, dataset):
        self.img_paths = {}
        prefix = f'{ROOT_IMAGE_DIR}/'
        for f in glob.glob(f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-GRiD-3D/grid-3d/part_*/*.png'):
            idx = int(os.path.splitext(os.path.basename(f))[0])
            self.img_paths[idx] = f[len(prefix):] if f.startswith(prefix) else f
        return super().prepare_dataset(dataset)

    def preprocess(self, row):
        img = self.img_paths.get(row['img_idx'])
        if not img:
            return None
        return {
            'messages': [{'role': 'user', 'content': ' '.join(row['question'])},
                         {'role': 'assistant', 'content': row['answer']}],
            'images': [img],
        }


class RefSpatialPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        msgs = _conv_to_messages(row['conversations'])
        # images field contains relative paths like '2D/image/filename.jpg'
        imgs = [f"{BASE}/Molmo2-ER-RefSpatial/{img}" for img in row.get('images', [])]
        if imgs and msgs:
            msgs[0]['content'] = '<image>' * len(imgs) + msgs[0]['content']
        return {
            'messages': msgs,
            'images': imgs,
        }


class RoboPointPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        mgs = _conv_to_messages(row['conversations'])
        return {
            'messages': mgs,
            'images': [f"{BASE}/Molmo2-ER-RoboPoint/images/{img}" for img in row.get('images', [])],
        }


class RoboVQAPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        mgs = _conv_to_messages(row['conversations'])
        for m in mgs:
            if isinstance(m['content'], str) and m["role"] == "user":
                m['content'] = "<video>" + m['content']
                break
        return {
            'messages': mgs,
            'videos': [f"{BASE}/Molmo2-ER-RoboVQA/{row['videos']}"],
        }


class SATPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        msgs = row['messages']
        imgs = [os.path.join(BASE, 'Molmo2-ER-SAT', p) for p in row['images']]
        if imgs and msgs:
            msgs[0]['content'] = '<image>' * len(imgs) + msgs[0]['content']
        return {'messages': msgs, 'images': imgs}


class SenseNovaLoader(DatasetLoader):
    def _load_dataset_path(self, dataset_path, dataset_meta):
        import json
        from datasets import Dataset as HfDataset

        def gen():
            with open(dataset_path) as f:
                for line in f:
                    row = json.loads(line)
                    img = row.get('image', [])
                    if isinstance(img, str):
                        row['image'] = [img]
                    yield row

        dataset = HfDataset.from_generator(gen, cache_dir=os.path.join(get_cache_dir(), 'datasets'))
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file,
            strict=self.strict, enable_auto_mapping=not self.disable_auto_column_mapping)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset


class SenseNovaPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        return {
            'messages': _conv_to_messages(row['conversations']),
            'images': [f"{BASE}/Molmo2-ER-SenseNova-SI/{p}" for p in row['images']],
        }


class SIMSVSILoader(DatasetLoader):
    def _load_dataset_path(self, dataset_path, dataset_meta):
        import pyarrow.parquet as pq
        from datasets import Dataset as HfDataset

        def gen():
            pf = pq.ParquetFile(dataset_path)
            for batch in pf.iter_batches(batch_size=1000):
                for row in batch.to_pylist():
                    row['conversations'] = [
                        {'from': c['from'], 'value': c['value']}
                        for c in row['conversations']
                    ]
                    row['images'] = row.get('images') or []
                    yield row

        dataset = HfDataset.from_generator(gen, cache_dir=os.path.join(get_cache_dir(), 'datasets'))
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file,
            strict=self.strict, enable_auto_mapping=not self.disable_auto_column_mapping)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset


class SIMSVSIPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        msgs = _conv_to_messages(row['conversations'])
        for m in msgs:
            if isinstance(m['content'], str):
                m['content'] = m['content'].replace('<image>', '<video>')
        return {
            'messages': msgs,
            'videos': [f"{BASE}/Molmo2-ER-SIMS-VSI/{row['videos']}"],
        }


class VSI590KLoader(DatasetLoader):
    def _load_dataset_path(self, dataset_path, dataset_meta):
        import json
        from datasets import Dataset as HfDataset, Features, Sequence, Value

        features = Features({
            'conversations': [{'from': Value('string'), 'value': Value('string')}],
            'images': Sequence(Value('string')),
            'videos': Sequence(Value('string')),
            'question_type': Value('string'),
        })

        def gen():
            with open(dataset_path) as f:
                for line in f:
                    row = json.loads(line)
                    row.setdefault('images', [])
                    row.setdefault('videos', [])
                    row.setdefault('question_type', '')
                    yield row

        dataset = HfDataset.from_generator(gen, features=features,
                                           cache_dir=os.path.join(get_cache_dir(), 'datasets'))
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file,
            strict=self.strict, enable_auto_mapping=not self.disable_auto_column_mapping)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset


class VSI590KPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        msgs = _conv_to_messages(row['conversations'])
        return {
            'messages': msgs,
            'images': [f"{BASE}/Molmo2-ER-VSI-590K/{img}" for img in row.get('images', [])],
            'videos': [f"{BASE}/Molmo2-ER-VSI-590K/{vid}" for vid in row.get('videos', [])],
        }


class VSTPPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        msgs = _conv_to_messages(row['conversations'])
        for m in msgs:
            if isinstance(m['content'], str):
                m['content'] = m['content'].replace('<|image_pad|>', '<image>')
        return {
            'messages': msgs,
            'images': [f"{BASE}/Molmo2-ER-VST-P/{img}" for img in row.get('images', [])],
        }


class SATv2Preprocessor(RowPreprocessor):
    """SAT-v2: abs image paths already set; inject <image> tags into user content."""
    def preprocess(self, row):
        msgs = list(row['messages'])
        imgs = row.get('images', [])
        if imgs and msgs:
            msgs[0] = dict(msgs[0])
            msgs[0]['content'] = '<image>' * len(imgs) + msgs[0]['content']
        return {'messages': msgs, 'images': imgs}


class EgoPlanPreprocessor(RowPreprocessor):
    """EgoPlan: one multi-timestamp video-frame ref -> N single-frame refs + N <image> tags."""
    def preprocess(self, row):
        msgs = list(row['messages'])
        ref = row['images'][0]
        rel, ts_str = ref.rsplit(':', 1)
        timestamps = ts_str.split(',')
        full = f'{BASE}/EgoPlan/{rel}'
        images = [f'{full}:{t}' for t in timestamps]
        if msgs:
            msgs[0] = dict(msgs[0])
            msgs[0]['content'] = '<image>' * len(images) + msgs[0]['content']
        return {'messages': msgs, 'images': images}


class MixERAllLoader(DatasetLoader):
    """Load all *.jsonl files under Mix-ER-All."""
    def _load_dataset_path(self, dataset_path, dataset_meta):
        import json
        from datasets import Dataset as HfDataset

        def gen():
            for jsonl_file in sorted(glob.glob(dataset_path)):
                with open(jsonl_file) as f:
                    for line in f:
                        yield json.loads(line)

        dataset = HfDataset.from_generator(gen, cache_dir=os.path.join(get_cache_dir(), 'datasets'))
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file,
            strict=self.strict, enable_auto_mapping=not self.disable_auto_column_mapping)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset


class MixERAllPreprocessor(RowPreprocessor):
    """Mix-ER-All: messages/images with abs paths; inject <image> tags if missing."""
    def preprocess(self, row):
        msgs = list(row['messages'])
        imgs = row.get('images', [])
        if imgs and msgs:
            first_content = msgs[0].get('content', '')
            if '<image>' not in first_content:
                msgs[0] = dict(msgs[0])
                msgs[0]['content'] = '<image>' * len(imgs) + first_content
        return {'messages': msgs, 'images': imgs}


ROOT_IMAGE_DIR = os.environ.get('ROOT_IMAGE_DIR')
register_dataset(DatasetMeta(
    dataset_name='molmo2-er-clevr',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-CLEVR/CLEVR_v1.0/questions/CLEVR_train_questions.json',
    preprocess_func=CLEVRPreprocessor(),
    loader=CLEVRLoader,
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-grid3d',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-GRiD-3D/grid-3d/questions.json',
    preprocess_func=GRiD3DPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-refspatial',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-RefSpatial/refspatial_merged.jsonl',
    preprocess_func=RefSpatialPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-robopoint',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-RoboPoint/robopoint_1432k_qwen.jsonl',
    preprocess_func=RoboPointPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-robovqa',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-RoboVQA/robovqa_*.json',
    preprocess_func=RoboVQAPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-sat',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-SAT/exported/SAT_train.jsonl',
    preprocess_func=SATPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-sensenova-si',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-SenseNova-SI/SenseNova-SI-800K.jsonl',
    preprocess_func=SenseNovaPreprocessor(),
    loader=SenseNovaLoader,
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-sims-vsi',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-SIMS-VSI/sims_vsi_200k.parquet',
    preprocess_func=SIMSVSIPreprocessor(),
    loader=SIMSVSILoader,
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-vsi-590k',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-VSI-590K/vsi_590k_clean.jsonl',
    preprocess_func=VSI590KPreprocessor(),
    loader=VSI590KLoader,
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-vstp',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/Molmo2-ER-VST-P/vst_500k.json',
    preprocess_func=VSTPPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-sat-v2',
    dataset_path='/cpfs01/cpfs01/datas/vlm_dataset/molmo2_er_datasets/SAT-v2/train.jsonl',
    preprocess_func=SATv2Preprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-egoplan',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/EgoPlan/EgoPlan_IT_messages.jsonl',
    preprocess_func=EgoPlanPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-egoplan-mc',
    dataset_path=f'{ROOT_IMAGE_DIR}/{BASE}/EgoPlan/EgoPlan_mc_messages.jsonl',
    preprocess_func=EgoPlanPreprocessor(),
))

register_dataset(DatasetMeta(
    dataset_name='molmo2-er-mix-er-all',
    dataset_path='/cpfs01/cpfs01/datas/vlm_dataset/molmo2_er_datasets/Mix-ER-All/*.jsonl',
    preprocess_func=MixERAllPreprocessor(),
    loader=MixERAllLoader,
))
