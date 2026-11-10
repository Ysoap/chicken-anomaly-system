"""Smoke test Stage 1 — artefak hasil training benar-benar ada dan konsisten.

Jalankan dari root repo:  python -m pytest tests/test_stage1.py -v
Test ini ringan (tidak butuh GPU / tidak me-load model): hanya memeriksa
artefak hasil training Stage 1 dan konfigurasi yang dipakai.
"""
import os

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1 = os.path.join(ROOT, 'backend', 'stage1_detection')
BEST = os.path.join(STAGE1, 'models', 'best.pth')
BEST_MD5 = '6f8457e56c9147d1ba8efc14a7ba05c2'
BEST_SIZE = 322403181


def test_best_pth_ada_dan_utuh():
    assert os.path.exists(BEST), 'models/best.pth tidak ditemukan (git lfs pull?)'
    assert os.path.getsize(BEST) == BEST_SIZE, \
        f'Ukuran best.pth {os.path.getsize(BEST)} != {BEST_SIZE}'
    # File LFS yang belum di-pull hanya berisi pointer ~130 byte.
    assert os.path.getsize(BEST) > 10_000_000, \
        'best.pth tampak seperti pointer LFS, jalankan: git lfs pull'


def test_config_sesuai_training_nyata():
    pytest.importorskip('yaml')
    with open(os.path.join(STAGE1, 'configs', 'rt_detr_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    assert cfg['model'] == 'rt_detr_v2_s'
    assert cfg['num_classes'] == 1
    assert cfg['training']['epoches'] == 50
    assert cfg['dataset']['split']['train'].startswith('5222')


def test_training_log_memuat_50_epoch():
    import re
    with open(os.path.join(STAGE1, 'training_log.txt')) as f:
        text = f.read()
    epochs = sorted(set(int(e) for e in re.findall(r'"epoch":\s*(\d+)', text)))
    assert epochs == list(range(50)), f'epoch unik di log: {epochs}'
    assert '0.926' in text, 'AP terbaik 0,9261 tidak ditemukan di log'


def test_contoh_prediksi_ada():
    examples = os.path.join(STAGE1, 'examples')
    pngs = [p for p in os.listdir(examples) if p.endswith('_pred.png')]
    assert len(pngs) >= 2, 'Contoh gambar prediksi tidak ditemukan'
