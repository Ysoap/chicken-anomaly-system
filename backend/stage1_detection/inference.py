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

Mode tracking: `python inference.py --track --frames-dir DIR --fps 10`
atau `--video FILE`. Deteksi per frame + ByteTrack (supervision 0.25.1);
hasil uji 10 Nov 2026 di 72 frame fCam16: 91 ID unik, median umur track
33 frame, 10/33 ID awal bertahan sampai frame 71 (ID switch masih sering
— lihat README). Parameter tracking bisa diubah lewat
--match-threshold / --track-buffer / --activation-threshold.
"""

# --- Mode tracking (ByteTrack) -------------------------------------------
# Tracking ditambahkan 10 Nov 2026: deteksi per frame + ByteTrack dari
# library `supervision` (0.25.1 — versi terakhir yang masih membawa tracker;
# supervision >= 0.26 menghapus modul trackers). Tidak ada tracker yang
# ditulis sendiri; ByteTrack di sini adalah port resmi algoritma ByteTrack
# (Zhang et al. 2022) yang dibundel supervision.
#   python inference.py --track --frames-dir /path/ke/frame/ --fps 10
#   python inference.py --track --video /path/ke/video.mp4 --fps 10
# Output: video beranotasi (box + Track ID) + tracks JSON per frame.
# ---------------------------------------------------------------------------
import argparse
import json
import os
import random
import subprocess
import sys
import zipfile

import numpy as np
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


def track_frames(model, post, frame_iter, out_video, out_tracks, out_examples,
                 threshold, fps, example_idx, match_threshold=0.8,
                 track_buffer=30, activation_threshold=0.25):
    """Deteksi + ByteTrack di urutan frame.

    frame_iter: iterable (frame_index, name, PIL RGB image).
    Mengembalikan ringkasan statistik tracking (dict).
    """
    import cv2
    from supervision import BoxAnnotator, ByteTrack, Detections, LabelAnnotator

    tracker = ByteTrack(track_activation_threshold=activation_threshold,
                        lost_track_buffer=track_buffer,
                        minimum_matching_threshold=match_threshold,
                        frame_rate=30,
                        minimum_consecutive_frames=1)
    box_annotator = BoxAnnotator(thickness=2)
    label_annotator = LabelAnnotator()

    writer = None
    tracks = []          # [{frame, name, tracks: [{track_id, bbox, score}]}]
    first_seen = {}      # track_id -> frame index pertama
    last_seen = {}       # track_id -> frame index terakhir
    n_det_total = 0

    for fi, name, img in frame_iter:
        w, h = img.size
        boxes, scores = predict(model, post, img, threshold)
        n_det_total += len(boxes)
        if boxes:
            det = Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                confidence=np.array(scores, dtype=np.float32),
                class_id=np.zeros(len(boxes), dtype=int))
        else:
            det = Detections.empty()
        tracked = tracker.update_with_detections(det)

        frame = np.array(img)  # RGB
        if len(tracked) > 0:
            labels = [f'#{int(t)} {s:.2f}'
                      for t, s in zip(tracked.tracker_id, tracked.confidence)]
            frame = box_annotator.annotate(scene=frame, detections=tracked)
            frame = label_annotator.annotate(scene=frame, detections=tracked,
                                             labels=labels)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f'Gagal membuka VideoWriter: {out_video}')
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if fi in example_idx:
            Image.fromarray(frame).save(
                os.path.join(out_examples, f'track_frame_{fi:03d}.png'))

        recs = []
        for (x1, y1, x2, y2), s, tid in zip(
                tracked.xyxy, tracked.confidence, tracked.tracker_id):
            tid = int(tid)
            recs.append({'track_id': tid,
                         'bbox': [float(x1), float(y1), float(x2), float(y2)],
                         'score': float(s)})
            first_seen.setdefault(tid, fi)
            last_seen[tid] = fi
        tracks.append({'frame': fi, 'name': name, 'tracks': recs})
        if (fi + 1) % 10 == 0 or fi == 0:
            print(f'  frame {fi + 1}: {len(recs)} track aktif, '
                  f'{len(first_seen)} ID unik sejauh ini', flush=True)

    writer.release()
    with open(out_tracks, 'w') as f:
        json.dump({'fps': fps, 'score_threshold': threshold,
                   'frames': tracks}, f, indent=2)

    n_frames = len(tracks)
    lifetimes = {tid: last_seen[tid] - first_seen[tid] + 1
                 for tid in first_seen}
    ids_after_first = sorted(t for t, f in first_seen.items() if f > 0)
    summary = {
        'n_frames': n_frames,
        'fps': fps,
        'score_threshold': threshold,
        'avg_detections_per_frame': round(n_det_total / max(n_frames, 1), 2),
        'unique_track_ids': len(first_seen),
        'ids_in_frame_0': sum(1 for f in first_seen.values() if f == 0),
        'new_ids_after_frame_0': len(ids_after_first),
        'new_ids_after_frame_0_list': ids_after_first,
        'tracks_alive_ge_10_frames': sum(1 for v in lifetimes.values() if v >= 10),
        'tracks_alive_ge_30_frames': sum(1 for v in lifetimes.values() if v >= 30),
        'median_track_lifetime_frames':
            float(np.median(list(lifetimes.values()))) if lifetimes else 0,
        'max_track_lifetime_frames': max(lifetimes.values()) if lifetimes else 0,
    }
    print('=== Ringkasan tracking ===')
    print(json.dumps(summary, indent=2))
    return summary


def iter_frames_dir(frames_dir):
    names = sorted(f for f in os.listdir(frames_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    for i, fname in enumerate(names):
        yield i, fname, Image.open(os.path.join(frames_dir, fname)).convert('RGB')


def iter_video(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Gagal membuka video: {video_path}')
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        yield i, f'frame_{i:04d}', img
        i += 1
    cap.release()


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
    ap.add_argument('--track', action='store_true',
                    help='Mode tracking: deteksi + ByteTrack di video/frame.')
    ap.add_argument('--video', default=None,
                    help='Path video input (mode --track).')
    ap.add_argument('--frames-dir', default=None,
                    help='Direktori urutan frame (mode --track, alternatif --video).')
    ap.add_argument('--fps', type=float, default=10,
                    help='FPS video output mode tracking (default 10).')
    ap.add_argument('--track-threshold', type=float, default=0.3,
                    help='Score threshold deteksi untuk tracking (default 0.3).')
    ap.add_argument('--match-threshold', type=float, default=0.8,
                    help='IoU minimum ByteTrack untuk match antar frame (default 0.8).')
    ap.add_argument('--track-buffer', type=int, default=30,
                    help='Buffer frame ByteTrack sebelum track dianggap hilang (default 30).')
    ap.add_argument('--activation-threshold', type=float, default=0.25,
                    help='Score minimum ByteTrack untuk track baru (default 0.25).')
    ap.add_argument('--track-out', default=os.path.join(HERE, 'examples', 'tracked.mp4'),
                    help='Path video beranotasi hasil tracking.')
    ap.add_argument('--tracks-json', default=os.path.join(HERE, 'results', 'tracks.json'),
                    help='Path JSON track ID per frame.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, post = load_model(args.best, args.repo_dir)

    if args.track:
        if bool(args.video) == bool(args.frames_dir):
            ap.error('Mode --track butuh tepat satu dari --video atau --frames-dir.')
        frames = (iter_video(args.video) if args.video
                  else iter_frames_dir(args.frames_dir))
        out_dir = os.path.dirname(os.path.abspath(args.track_out))
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.tracks_json)),
                    exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
        n_frames = (sum(1 for _ in iter_video(args.video)) if args.video
                    else len([f for f in os.listdir(args.frames_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]))
        example_idx = {0, n_frames // 3, 2 * n_frames // 3, n_frames - 1}
        summary = track_frames(model, post, frames, args.track_out,
                               args.tracks_json, args.out_dir,
                               args.track_threshold, args.fps, example_idx,
                               match_threshold=args.match_threshold,
                               track_buffer=args.track_buffer,
                               activation_threshold=args.activation_threshold)
        with open(os.path.join(os.path.dirname(args.tracks_json),
                               'tracking_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('Video beranotasi:', args.track_out)
        print('Tracks JSON:', args.tracks_json)
        return

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
