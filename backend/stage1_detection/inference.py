"""Stage 1 — inference deteksi ayam dari best.pth (RT-DETRv2-S).

Menjalankan models/best.pth pada N gambar Validation acak, menggambar box
prediksi (merah + skor) dan box ground-truth (hijau) di atas gambar asli,
lalu menyimpan hasilnya sebagai PNG. Dipakai untuk cek visual bahwa box
deteksi masuk akal sebelum lanjut ke Stage 2 (tracking + perilaku).

Detail yang sudah diverifikasi (10 Nov 2026):
  - Bobot yang dipakai adalah bobot EMA dari checkpoint (ckpt['ema']['module'])
    — sama seperti yang dipakai saat evaluasi AP 0,9261.
  - Preprocessing sama seperti validasi saat training: resize 640x640 +
    skala ke [0, 1] (tanpa normalisasi mean/std).
  - Postprocessor RT-DETR resmi; orig_size yang diteruskan adalah [w, h]
    (sesuai src/data/dataset/coco_dataset.py: target["orig_size"] = [w, h]).
    JANGAN dibalik jadi [h, w] — box jadi kurus-memanjang (bug ini sempat
    terjadi dan sudah diperbaiki).

Hasil cek 10 Nov 2026 (5 gambar Validation acak, threshold 0,5):
  fCam16.20230220_214500_0011.jpg: GT 33, prediksi 33 (skor 0,56-0,99)
  fCam16.20230220_214500_0071.jpg: GT 34, prediksi 34 (skor 0,93-0,99)
  fCam16.20230221_130001_0045.jpg: GT 33, prediksi 33 (skor 0,90-0,99)
  frame_01_2.49s.jpg:              GT 10, prediksi 10 (skor 0,93-0,98)
  frame_42_615.36s.jpg:            GT 8,  prediksi 8  (skor 0,70-0,94)
  Catatan: di frame_42 (anak ayam kecil di bawah lampu merah) ada 1 ayam kecil
  yang sebagian tertutup terlewat + 1 false positive di terpal (skor 0,70).

Contoh pakai:
  python inference.py --zip /path/ke/ChickenDet.zip
  python inference.py --zip /path/ke/ChickenDet.zip --samples 3 --threshold 0.6

Catatan: tracking (ByteTrack) BELUM terintegrasi di script ini — ini murni
deteksi per frame. ByteTrack adalah langkah berikut Stage 1.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import zipfile

import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = 'https://github.com/lyuwenyu/RT-DETR.git'
CONFIG_REL = os.path.join('configs', 'rtdetrv2', 'rtdetrv2_r18vd_120e_coco.yml')


def load_model(best_path, repo_dir):
    sys.path.insert(0, repo_dir)
    if not os.path.exists(os.path.join(repo_dir, 'tools', 'train.py')):
        print('Clone repo resmi RT-DETR...')
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL,
                        os.path.dirname(repo_dir)], check=True)
    from src.core import YAMLConfig  # noqa: E402

    config = os.path.join(repo_dir, CONFIG_REL)
    cfg = YAMLConfig(config, num_classes=1, remap_mscoco_category=True)
    model = cfg.model
    post = cfg.postprocessor
    print('Memuat checkpoint:', best_path, os.path.getsize(best_path), 'byte')
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    print('Isi checkpoint:', sorted(ckpt.keys()), '| last_epoch:', ckpt.get('last_epoch'))
    if 'ema' in ckpt and isinstance(ckpt['ema'], dict) and 'module' in ckpt['ema']:
        state = ckpt['ema']['module']
        print('Pakai bobot EMA (sama seperti evaluasi AP).')
    else:
        state = ckpt['model']
        print('Pakai bobot model (tanpa EMA).')
    model.load_state_dict(state)
    model.eval()
    return model, post


def pick_samples(zip_path, n_samples, seed):
    z = zipfile.ZipFile(zip_path)
    val_imgs = [n for n in z.namelist()
                if n.startswith('images/Validation/') and n.lower().endswith('.jpg')]
    random.seed(seed)
    pick = sorted(random.sample(val_imgs, min(n_samples, len(val_imgs))))
    ann = json.loads(z.read('annotations/instances_Validation.json'))
    img_by_name = {im['file_name']: im['id'] for im in ann['images']}
    gt = {}
    for a in ann['annotations']:
        gt.setdefault(a['image_id'], []).append(a['bbox'])  # [x, y, w, h]
    return z, pick, img_by_name, gt


def predict(model, post, img, threshold):
    import numpy as np
    w, h = img.size
    small = img.resize((640, 640))
    tensor = torch.from_numpy(np.array(small)).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0))
    results = post(outputs, torch.tensor([[w, h]]))  # orig_size = [w, h]
    r = results[0]
    keep = r['scores'] >= threshold
    return r['boxes'][keep].tolist(), r['scores'][keep].tolist()


def draw(img, boxes, scores, gt_boxes, out_path, title, threshold):
    d = ImageDraw.Draw(img)
    for (x, y, bw, bh) in gt_boxes:
        d.rectangle([x, y, x + bw, y + bh], outline=(0, 200, 0), width=3)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        d.text((x1 + 2, y1 + 2), f'{s:.2f}', fill=(255, 0, 0))
    img.save(out_path)
    print(f'{title}: GT={len(gt_boxes)} box, prediksi>={threshold}: {len(boxes)} box, '
          f'skor={[round(s, 2) for s in scores]} -> {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--best', default=os.path.join(HERE, 'models', 'best.pth'))
    ap.add_argument('--zip', default=None,
                    help='Path ChickenDet.zip (untuk gambar + GT Validation). '
                         'Bila tidak diisi, pakai --images (tanpa GT).')
    ap.add_argument('--images', nargs='+', default=None,
                    help='Daftar path gambar (alternatif --zip, tanpa GT).')
    ap.add_argument('--repo-dir', default=os.path.join(HERE, 'work', 'RT-DETR',
                                                       'rtdetrv2_pytorch'))
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'examples'))
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, post = load_model(args.best, args.repo_dir)

    if args.zip:
        z, pick, img_by_name, gt = pick_samples(args.zip, args.samples, args.seed)
        import io
        for name in pick:
            fname = os.path.basename(name)
            img = Image.open(io.BytesIO(z.read(name))).convert('RGB')
            boxes, scores = predict(model, post, img, args.threshold)
            draw(img, boxes, scores, gt.get(img_by_name[fname], []),
                 os.path.join(args.out_dir, fname.rsplit('.', 1)[0] + '_pred.png'),
                 fname, args.threshold)
        z.close()
    elif args.images:
        for path in args.images:
            img = Image.open(path).convert('RGB')
            boxes, scores = predict(model, post, img, args.threshold)
            out = os.path.join(args.out_dir,
                               os.path.basename(path).rsplit('.', 1)[0] + '_pred.png')
            draw(img, boxes, scores, [], out, os.path.basename(path), args.threshold)
    else:
        ap.error('Isi --zip atau --images.')
    print('Selesai. Hasil di:', args.out_dir)


if __name__ == '__main__':
    main()
