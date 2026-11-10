"""Stage 1 — evaluasi best.pth (RT-DETRv2-S) di test set ChickenDet (250 gambar).

Test set = kandang komersial held-out, tidak pernah dipakai training maupun
validasi. Skrip ini menjalankan inference di semua 250 gambar test, lalu
menghitung AP/AP50 dengan pycocotools COCOeval terhadap
annotations/instances_Test.json (iscrowd diperbaiki 1 -> 0, sama seperti
saat training — lihat train.py prepare_dataset).

Bobot yang dipakai: EMA dari checkpoint (ckpt['ema']['module']), sama seperti
evaluasi val (AP 0,9261). Preprocessing sama seperti validasi: resize 640x640
+ skala [0, 1], tanpa normalisasi mean/std. Postprocessor resmi RT-DETR;
orig_size = [w, h] (BUKAN [h, w] — lihat inference.py).

Contoh pakai (dari root repo):
  python backend/stage1_detection/eval_test_set.py \
      --zip /path/ke/ChickenDet.zip \
      --repo-dir /path/ke/RT-DETR/rtdetrv2_pytorch

Hasil disimpan sebagai JSON di backend/stage1_detection/results/test_set_eval.json
(metrik COCO + metadata run: tanggal, threshold, jumlah gambar/prediksi).
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from datetime import date

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from inference import load_model, predict  # noqa: E402

DEFAULT_REPO_DIR = os.path.join(HERE, 'work', 'RT-DETR', 'rtdetrv2_pytorch')
DEFAULT_OUT = os.path.join(HERE, 'results', 'test_set_eval.json')


def load_test_gt(zip_path):
    """Baca instances_Test.json dari zip; perbaiki iscrowd 1 -> 0.

    Mengembalikan (coco_gt_dict, img_by_name) — coco_gt_dict dipakai
    langsung untuk pycocotools COCO (tanpa menulis file *_fixed.json).
    """
    z = zipfile.ZipFile(zip_path)
    gt = json.loads(z.read('annotations/instances_Test.json'))
    n_crowd = 0
    for a in gt['annotations']:
        if a.get('iscrowd'):
            a['iscrowd'] = 0
            n_crowd += 1
    img_by_name = {im['file_name']: im for im in gt['images']}
    print(f'GT test: {len(gt["images"])} gambar, {len(gt["annotations"])} '
          f'anotasi, iscrowd diperbaiki: {n_crowd}')
    return z, gt, img_by_name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--best', default=os.path.join(HERE, 'models', 'best.pth'))
    ap.add_argument('--zip', required=True, help='Path ChickenDet.zip')
    ap.add_argument('--repo-dir', default=DEFAULT_REPO_DIR,
                    help='Path clone repo resmi RT-DETR (rtdetrv2_pytorch)')
    ap.add_argument('--threshold', type=float, default=0.05,
                    help='Score threshold prediksi (rendah, untuk COCOeval)')
    ap.add_argument('--max-boxes', type=int, default=300,
                    help='Maks box prediksi per gambar')
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.time()
    model, post = load_model(args.best, args.repo_dir)
    z, gt, img_by_name = load_test_gt(args.zip)

    test_imgs = sorted(n for n in z.namelist()
                       if n.startswith('images/Test/') and n.lower().endswith('.jpg'))
    assert len(test_imgs) == 250, f'Jumlah gambar test: {len(test_imgs)} (harusnya 250)'
    assert len(gt['images']) == 250

    results = []
    n_pred_total = 0
    for i, name in enumerate(test_imgs):
        fname = os.path.basename(name)
        img = Image.open(io.BytesIO(z.read(name))).convert('RGB')
        boxes, scores = predict(model, post, img, args.threshold)
        order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        boxes = [boxes[k] for k in order[:args.max_boxes]]
        scores = [scores[k] for k in order[:args.max_boxes]]
        image_id = img_by_name[fname]['id']
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            results.append({
                'image_id': image_id,
                'category_id': 1,
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(s),
            })
        n_pred_total += len(boxes)
        if (i + 1) % 25 == 0 or i == 0:
            print(f'[{i + 1}/250] {fname}: {len(boxes)} box, '
                  f'elapsed {time.time() - t0:.0f}s', flush=True)
    z.close()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
    ev.evaluate()
    ev.accumulate()
    print('=== COCOeval test set ===')
    ev.summarize()

    stats = ev.stats
    out = {
        'model': 'rt_detr_v2_s (RT-DETRv2-S, rtdetrv2_r18vd_120e_coco)',
        'checkpoint': 'models/best.pth (bobot EMA)',
        'checkpoint_md5': '6f8457e56c9147d1ba8efc14a7ba05c2',
        'date': date.today().isoformat(),
        'split': 'test (held-out, 250 gambar)',
        'score_threshold': args.threshold,
        'max_boxes_per_image': args.max_boxes,
        'n_images': 250,
        'n_gt_annotations': len(gt['annotations']),
        'n_predictions': n_pred_total,
        'elapsed_seconds': round(time.time() - t0, 1),
        'AP': round(float(stats[0]), 4),
        'AP50': round(float(stats[1]), 4),
        'AP75': round(float(stats[2]), 4),
        'AP_small': round(float(stats[3]), 4),
        'AP_medium': round(float(stats[4]), 4),
        'AP_large': round(float(stats[5]), 4),
        'AR_max1': round(float(stats[6]), 4),
        'AR_max10': round(float(stats[7]), 4),
        'AR_max100': round(float(stats[8]), 4),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Hasil disimpan: {args.out}')
    print(json.dumps({k: out[k] for k in
                      ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large')},
                     indent=2))


if __name__ == '__main__':
    main()
