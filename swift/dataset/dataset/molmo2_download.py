"""Molmo2 Dataset Download Engine.

Supports resume, parallel downloads, and retry for failed items.

Usage:
    from swift.dataset.dataset.molmo2_download import Molmo2Downloader

    downloader = Molmo2Downloader('/cpfs01/cpfs01/datas/molmo2_datasets', max_workers=16)
    downloader.download('Molmo2-Cap')
    downloader.download_all()
    downloader.status('Molmo2-Cap')
"""
import glob
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Datasets that have youtube_id_to_urls_mapping.json with GCP URLs
YOUTUBE_MAPPED_DATASETS = [
    'Molmo2-AskModelAnything',
    'Molmo2-VideoCapQA',
    'Molmo2-VideoPoint',
    'Molmo2-VideoPointEval',
    'Molmo2-VideoSubtitleQA',
]

# Datasets with video_id but NO mapping file (YouTube-only, need yt-dlp or manual)
YOUTUBE_ONLY_DATASETS = [
    'Molmo2-Cap',
    'Molmo2-VideoCountEval',
]

# Datasets that use image_urls column
IMAGE_URL_DATASETS = [
    'Molmo2-MultiImagePoint',
    'Molmo2-MultiImageQA',
]

# Datasets with no download needed (embedded bytes or external sources)
NO_DOWNLOAD_DATASETS = [
    'Molmo2-SynMultiImageQA',
    'Molmo2-VideoTrack',
    'Molmo2-VideoTrackEval',
]

@dataclass
class DownloadStatus:
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: List[str] = field(default_factory=list)


def _download_file(url: str, dest: str, timeout: int = 300, retries: int = 3) -> bool:
    """Download a single file with resume support and retry."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    for attempt in range(retries):
        try:
            headers = {}
            mode = 'wb'
            existing_size = 0

            if os.path.exists(dest):
                existing_size = os.path.getsize(dest)
                head = requests.head(url, timeout=30, allow_redirects=True)
                if head.status_code == 200:
                    remote_size = int(head.headers.get('Content-Length', 0))
                    if remote_size > 0 and existing_size == remote_size:
                        return True  # already complete
                    if remote_size > 0 and existing_size < remote_size:
                        headers['Range'] = f'bytes={existing_size}-'
                        mode = 'ab'
                    else:
                        existing_size = 0

            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            if resp.status_code == 416:
                return True  # range not satisfiable = file complete
            resp.raise_for_status()

            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return True

        except (requests.RequestException, IOError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f'Failed to download {url} -> {dest}: {e}')
                return False
    return False


class VideoDownloader:
    """Downloads videos from GCP URLs using youtube_id_to_urls_mapping.json."""

    def __init__(self, base_dir: str, dataset_name: str):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(base_dir, dataset_name)
        self.video_dir = os.path.join(self.dataset_dir, 'videos')
        os.makedirs(self.video_dir, exist_ok=True)

    def _load_mapping(self) -> Dict[str, dict]:
        mapping_path = os.path.join(self.dataset_dir, 'youtube_id_to_urls_mapping.json')
        with open(mapping_path) as f:
            return json.load(f)

    def _get_needed_video_ids(self) -> Set[str]:
        parquets = glob.glob(os.path.join(self.dataset_dir, 'data', '*.parquet'))
        video_ids = set()
        for p in parquets:
            df = pd.read_parquet(p, columns=['video_id'])
            video_ids.update(df['video_id'].dropna().unique())
        return video_ids

    def download(self, max_workers: int = 8, limit: Optional[int] = None,
                 overwrite: bool = False) -> DownloadStatus:
        mapping = self._load_mapping()
        needed_ids = self._get_needed_video_ids()
        if limit:
            needed_ids = set(list(needed_ids)[:limit])

        tasks = []
        for vid_id in needed_ids:
            dest = os.path.join(self.video_dir, f'{vid_id}.mp4')
            if not overwrite and os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            if vid_id in mapping:
                url = mapping[vid_id].get('gcp_url') or mapping[vid_id].get('youtube_url')
                if url:
                    tasks.append((url, dest, vid_id))

        status = DownloadStatus(total=len(needed_ids), skipped=len(needed_ids) - len(tasks))

        if not tasks:
            status.downloaded = status.skipped
            return status

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_file, url, dest): vid_id
                       for url, dest, vid_id in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f'Downloading {self.dataset_name} videos'):
                vid_id = futures[future]
                if future.result():
                    status.downloaded += 1
                else:
                    status.failed.append(vid_id)

        status.downloaded += status.skipped
        self._save_failures(status.failed)
        return status

    def _save_failures(self, failed: List[str]):
        path = os.path.join(self.dataset_dir, '.download_failures.json')
        if failed:
            with open(path, 'w') as f:
                json.dump(failed, f)
        elif os.path.exists(path):
            os.remove(path)


class ImageDownloader:
    """Downloads images from HTTP URLs for MultiImagePoint/MultiImageQA."""

    def __init__(self, base_dir: str, dataset_name: str):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(base_dir, dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

    def _get_url_sha_pairs(self) -> List[Tuple[str, str]]:
        parquet_dirs = [
            os.path.join(self.dataset_dir, 'data'),
            os.path.join(self.dataset_dir, 'parquet'),
        ]
        pairs = set()
        for d in parquet_dirs:
            for p in glob.glob(os.path.join(d, '*.parquet')):
                df = pd.read_parquet(p, columns=['image_urls', 'image_sha256s'])
                for _, row in df.iterrows():
                    urls = row['image_urls']
                    shas = row['image_sha256s']
                    if urls is None:
                        continue
                    for url, sha in zip(urls, shas):
                        ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
                        pairs.add((url, sha, ext))
        return sorted(pairs, key=lambda x: x[1])

    def download(self, max_workers: int = 16, limit: Optional[int] = None,
                 overwrite: bool = False, delay: float = 0.05) -> DownloadStatus:
        all_pairs = self._get_url_sha_pairs()
        if limit:
            all_pairs = all_pairs[:limit]

        tasks = []
        for url, sha, ext in all_pairs:
            dest = os.path.join(self.image_dir, f'{sha}{ext}')
            if not overwrite and os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            tasks.append((url, dest, sha))

        status = DownloadStatus(total=len(all_pairs), skipped=len(all_pairs) - len(tasks))

        if not tasks:
            status.downloaded = status.skipped
            return status

        def _download_with_delay(args):
            url, dest, _ = args
            if delay > 0:
                time.sleep(delay)
            return _download_file(url, dest, timeout=60)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_with_delay, t): t[2] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f'Downloading {self.dataset_name} images'):
                sha = futures[future]
                if future.result():
                    status.downloaded += 1
                else:
                    status.failed.append(sha)

        status.downloaded += status.skipped
        self._save_failures(status.failed)
        return status

    def _save_failures(self, failed: List[str]):
        path = os.path.join(self.dataset_dir, '.download_failures.json')
        if failed:
            with open(path, 'w') as f:
                json.dump(failed, f)
        elif os.path.exists(path):
            os.remove(path)


class VimeoExtractor:
    """Extracts vimeo_videos.zip for CapEval dataset."""

    def __init__(self, base_dir: str):
        self.dataset_dir = os.path.join(base_dir, 'Molmo2-CapEval')
        self.zip_path = os.path.join(self.dataset_dir, 'vimeo_videos.zip')
        self.video_dir = os.path.join(self.dataset_dir, 'videos')

    def extract(self) -> DownloadStatus:
        if not os.path.exists(self.zip_path):
            logger.warning(f'vimeo_videos.zip not found at {self.zip_path}')
            return DownloadStatus(total=1, failed=['vimeo_videos.zip'])

        if os.path.exists(self.video_dir) and os.listdir(self.video_dir):
            return DownloadStatus(total=1, downloaded=1, skipped=1)

        os.makedirs(self.video_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            zf.extractall(self.video_dir)
        return DownloadStatus(total=1, downloaded=1)


class Molmo2Downloader:
    """Unified download engine for all Molmo2 datasets."""

    def __init__(self, base_dir: str = '/cpfs01/cpfs01/datas/molmo2_datasets',
                 max_workers: int = 8):
        self.base_dir = base_dir
        self.max_workers = max_workers

    def download(self, dataset_name: str, max_workers: Optional[int] = None,
                 limit: Optional[int] = None, overwrite: bool = False,
                 retry_failed: bool = False) -> DownloadStatus:
        """Download media for a specific dataset.

        Args:
            dataset_name: Name of the dataset folder (e.g. 'Molmo2-Cap')
            max_workers: Number of parallel download threads
            limit: Max number of files to download (for testing)
            overwrite: Re-download existing files
            retry_failed: Only retry previously failed downloads
        """
        workers = max_workers or self.max_workers

        if dataset_name in NO_DOWNLOAD_DATASETS:
            logger.info(f'{dataset_name}: no download needed (embedded/external)')
            return DownloadStatus()

        if dataset_name in YOUTUBE_ONLY_DATASETS:
            logger.warning(f'{dataset_name}: no mapping file available. '
                           f'Videos must be downloaded manually via yt-dlp or provided externally.')
            return DownloadStatus()

        if dataset_name == 'Molmo2-CapEval':
            extractor = VimeoExtractor(self.base_dir)
            vimeo_status = extractor.extract()
            # CapEval also has YouTube videos via mapping
            mapping_path = os.path.join(self.base_dir, dataset_name, 'youtube_id_to_urls_mapping.json')
            if os.path.exists(mapping_path):
                vid_dl = VideoDownloader(self.base_dir, dataset_name)
                vid_status = vid_dl.download(max_workers=workers, limit=limit, overwrite=overwrite)
                vid_status.total += vimeo_status.total
                vid_status.downloaded += vimeo_status.downloaded
                return vid_status
            return vimeo_status

        if dataset_name in YOUTUBE_MAPPED_DATASETS:
            dl = VideoDownloader(self.base_dir, dataset_name)
            return dl.download(max_workers=workers, limit=limit, overwrite=overwrite)

        if dataset_name in IMAGE_URL_DATASETS:
            dl = ImageDownloader(self.base_dir, dataset_name)
            return dl.download(max_workers=workers, limit=limit, overwrite=overwrite)

        logger.warning(f'Unknown dataset: {dataset_name}')
        return DownloadStatus()

    def download_all(self, max_workers: Optional[int] = None,
                     limit: Optional[int] = None) -> Dict[str, DownloadStatus]:
        """Download all downloadable datasets."""
        results = {}
        all_datasets = YOUTUBE_MAPPED_DATASETS + IMAGE_URL_DATASETS + ['Molmo2-CapEval']
        for name in all_datasets:
            logger.info(f'Starting download: {name}')
            results[name] = self.download(name, max_workers=max_workers, limit=limit)
            logger.info(f'{name}: {results[name].downloaded}/{results[name].total} done, '
                        f'{len(results[name].failed)} failed')
        return results

    def status(self, dataset_name: str) -> Dict:
        """Check download status for a dataset."""
        dataset_dir = os.path.join(self.base_dir, dataset_name)

        if dataset_name in YOUTUBE_MAPPED_DATASETS or dataset_name == 'Molmo2-CapEval':
            video_dir = os.path.join(dataset_dir, 'videos')
            downloaded = len(glob.glob(os.path.join(video_dir, '*.mp4'))) if os.path.exists(video_dir) else 0
            parquets = glob.glob(os.path.join(dataset_dir, 'data', '*.parquet'))
            total = 0
            for p in parquets:
                df = pd.read_parquet(p, columns=['video_id'])
                total += df['video_id'].nunique()
            failures_path = os.path.join(dataset_dir, '.download_failures.json')
            failed = json.load(open(failures_path)) if os.path.exists(failures_path) else []
            return {'total': total, 'downloaded': downloaded, 'failed': failed}

        if dataset_name in IMAGE_URL_DATASETS:
            image_dir = os.path.join(dataset_dir, 'images')
            downloaded = len(os.listdir(image_dir)) if os.path.exists(image_dir) else 0
            return {'total': '(run download to compute)', 'downloaded': downloaded}

        return {'status': 'no download needed or unknown dataset'}

