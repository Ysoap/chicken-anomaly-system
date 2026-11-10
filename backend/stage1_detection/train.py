"""Stage 1 — training deteksi ayam (RT-DETRv2-S) pada ChickenDet.

Port standalone dari notebook Colab yang dipakai untuk training sebenarnya
(lihat notebooks/Colab-ChickenDet-RTDETRv2-Official.ipynb — notebook itu yang
dijalankan di Colab T4, 1-10 Nov 2026). Script ini menjalankan langkah yang
sama di mesin sendiri / Colab tanpa Drive:

  1. Unzip ChickenDet.zip (ChickenVerse) bila belum ada.
  2. PERBAIKAN WAJIB: ubah iscrowd 1 -> 0 di anotasi train/val (file asli
     tidak diubah, hasil disimpan sebagai *_fixed.json). Tanpa ini training
     jalan dengan 0 box karena kode dataset resmi membuang anotasi iscrowd=1.
  3. Clone repo resmi lyuwenyu/RT-DETR + download checkpoint pretrained COCO.
  4. Training (tuning dari COCO, atau resume dari last.pth bila ada) dengan
     thread sync checkpoint tiap 120 detik ke --drive-run-dir bila diisi.

Hasil training sebenarnya (10 Nov 2026):
  50 epoch (0-49), AP terbaik 0,9261 (epoch 49), AP50 0,9876,
  train_loss 3,6517. Model final: models/best.pth (bobot EMA).

Contoh pakai:
  python train.py --zip /path/ke/ChickenDet.zip --work-dir ./work
  python train.py --zip /path/ke/ChickenDet.zip --work-dir ./work \
      --drive-run-dir /content/drive/MyDrive/PBL-Ayam/runs/chickendet_rtdetrv2_s_official
"""
import argparse
import collections
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import zipfile

REPO_URL = 'https://github.com/lyuwenyu/RT-DETR.git'
PRETRAINED_URL = ('https://github.com/lyuwenyu/storage/releases/download/v0.2/'
                  'rtdetrv2_r18vd_120e_coco_rerun_48.1.pth')
CONFIG_REL = 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
RUN_NAME = 'chickendet_rtdetrv2_s_official'


def log(msg):
    print(msg, flush=True)


def prepare_dataset(zip_path, extract_dir):
    """Unzip + perbaiki iscrowd. Mengembalikan (train_img, train_ann, val_img, val_ann)."""
    if (not os.path.exists(extract_dir)
            or len(list(pathlib.Path(extract_dir).rglob('*.jpg'))) < 100):
        log('Unzip dimulai...')
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        log('Unzip selesai.')
    else:
        log('Folder hasil unzip sudah ada, dilewati.')

    ann_dir = os.path.join(extract_dir, 'annotations')
    for name in ('instances_Train.json', 'instances_Validation.json'):
        src = os.path.join(ann_dir, name)
        dst = os.path.join(ann_dir, name.replace('.json', '_fixed.json'))
        if os.path.exists(dst):
            log(f'{dst} sudah ada, dilewati.')
            continue
        with open(src) as f:
            d = json.load(f)
        before = collections.Counter(a.get('iscrowd') for a in d['annotations'])
        for a in d['annotations']:
            a['iscrowd'] = 0
        with open(dst, 'w') as f:
            json.dump(d, f)
        log(f"{name}: images={len(d['images'])} ann={len(d['annotations'])} "
            f"iscrowd {dict(before)} -> 0, tersimpan di {dst}")
    return (
        os.path.join(extract_dir, 'images', 'Train'),
        os.path.join(ann_dir, 'instances_Train_fixed.json'),
        os.path.join(extract_dir, 'images', 'Validation'),
        os.path.join(ann_dir, 'instances_Validation_fixed.json'),
    )


def ensure_repo(repo_dir, pretrained_path):
    if not os.path.exists(os.path.join(repo_dir, 'tools', 'train.py')):
        log('Clone repo resmi RT-DETR...')
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL,
                        os.path.dirname(repo_dir)], check=True)
    else:
        log('Repo sudah ada, clone dilewati.')
    if not (os.path.exists(pretrained_path)
            and os.path.getsize(pretrained_path) > 10_000_000):
        log('Download checkpoint pretrained COCO...')
        subprocess.run(['wget', '-O', pretrained_path, PRETRAINED_URL], check=True)
    else:
        log('Checkpoint pretrained sudah ada.')
    assert os.path.getsize(pretrained_path) > 10_000_000, 'Checkpoint pretrained gagal.'
    return os.path.join(repo_dir, CONFIG_REL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', required=True, help='Path ChickenDet.zip (ChickenVerse)')
    ap.add_argument('--work-dir', default='./work',
                    help='Folder kerja lokal (dataset hasil unzip + output)')
    ap.add_argument('--output-dir', default=None,
                    help='Folder output training (default: <work-dir>/rtdetrv2_output/<RUN_NAME>)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--drive-run-dir', default=None,
                    help='Opsional: folder run di Drive untuk sync checkpoint tiap 120 detik')
    ap.add_argument('--no-sync', action='store_true',
                    help='Matikan thread sync (bila --drive-run-dir tidak dipakai, sync memang mati)')
    args = ap.parse_args()

    assert os.path.exists(args.zip) and os.path.getsize(args.zip) > 3_000_000_000, \
        'ChickenDet.zip tidak ditemukan / terlalu kecil.'
    work = os.path.abspath(args.work_dir)
    extract_dir = os.path.join(work, 'ChickenDet')
    local_output = args.output_dir or os.path.join(work, 'rtdetrv2_output', RUN_NAME)
    repo_dir = os.path.join(work, 'RT-DETR', 'rtdetrv2_pytorch')
    pretrained = os.path.join(work, 'rtdetrv2_s_pretrained.pth')
    os.makedirs(local_output, exist_ok=True)

    train_img, train_ann, val_img, val_ann = prepare_dataset(args.zip, extract_dir)
    for p in (train_img, train_ann, val_img, val_ann):
        assert os.path.exists(p), f'Path tidak ada: {p}'
    config = ensure_repo(repo_dir, pretrained)

    # --- Resume: last.pth di output lokal (atau Drive bila --drive-run-dir diisi) ---
    drive_run_dir = args.drive_run_dir
    local_last = os.path.join(local_output, 'last.pth')
    resume_mode = False
    if drive_run_dir:
        os.makedirs(drive_run_dir, exist_ok=True)
        drive_last = os.path.join(drive_run_dir, 'last.pth')
        if os.path.exists(drive_last) and os.path.getsize(drive_last) > 10_000_000:
            log('Ditemukan last.pth di Drive, resume dari situ...')
            shutil.copyfile(drive_last, local_last)
            drive_log = os.path.join(drive_run_dir, 'log.txt')
            if os.path.exists(drive_log):
                shutil.copyfile(drive_log, os.path.join(local_output, 'log.txt'))
            resume_mode = True
    if not resume_mode and os.path.exists(local_last) \
            and os.path.getsize(local_last) > 10_000_000:
        log('Ditemukan last.pth lokal, resume dari situ...')
        resume_mode = True

    # --- Thread sync checkpoint (hanya bila --drive-run-dir diisi) ---
    sync_stop = threading.Event()
    sync_seen = {}

    def copy_if_stable(src, dst):
        try:
            if not os.path.exists(src):
                return False
            s1 = os.path.getsize(src)
            time.sleep(2)
            if os.path.getsize(src) != s1:
                return False
            if sync_seen.get(src) == (s1, os.path.getmtime(src)):
                return False
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + '.tmp'
            shutil.copyfile(src, tmp)
            try:
                os.replace(tmp, dst)
            except OSError:
                shutil.copyfile(src, dst)
            sync_seen[src] = (s1, os.path.getmtime(src))
            log(f'[sync] {os.path.basename(src)} -> {dst} ({s1} byte)')
            return True
        except Exception as e:
            log(f'[sync] gagal salin {src}: {e}')
            return False

    def sync_loop():
        while not sync_stop.is_set():
            for name in ('last.pth', 'best.pth', 'log.txt'):
                copy_if_stable(os.path.join(local_output, name),
                               os.path.join(drive_run_dir, name))
            sync_stop.wait(120)

    sync_thread = None
    if drive_run_dir and not args.no_sync:
        sync_thread = threading.Thread(target=sync_loop, daemon=True)
        sync_thread.start()
        log('Sync checkpoint aktif (cek tiap 120 detik, hanya last.pth/best.pth/log.txt).')

    cmd = [sys.executable, os.path.join(repo_dir, 'tools', 'train.py'),
           '-c', config, '--use-amp', '--output-dir', local_output,
           '-u',
           f'epoches={args.epochs}',
           'num_classes=1',
           'remap_mscoco_category=True',
           'checkpoint_freq=5',
           'train_dataloader.total_batch_size=16',
           'val_dataloader.total_batch_size=32',
           f'train_dataloader.dataset.img_folder={train_img}',
           f'train_dataloader.dataset.ann_file={train_ann}',
           f'val_dataloader.dataset.img_folder={val_img}',
           f'val_dataloader.dataset.ann_file={val_ann}']
    if resume_mode:
        cmd += ['-r', local_last]
        log('Mode: RESUME dari last.pth')
    else:
        cmd += ['-t', pretrained]
        log('Mode: TUNING dari checkpoint COCO resmi')
    log('Perintah: ' + ' '.join(cmd))

    try:
        proc = subprocess.Popen(cmd, cwd=repo_dir, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='')
        proc.wait()
        log(f'Training process exit code: {proc.returncode}')
    finally:
        if sync_thread is not None:
            sync_stop.set()
            sync_thread.join(timeout=30)
            for name in ('last.pth', 'best.pth', 'log.txt'):
                copy_if_stable(os.path.join(local_output, name),
                               os.path.join(drive_run_dir, name))
            log('Sync checkpoint terakhir selesai.')

    log('Training selesai / sesi ini berakhir.')
    log('Best lokal: ' + os.path.join(local_output, 'best.pth'))
    if drive_run_dir:
        log('Best Drive: ' + os.path.join(drive_run_dir, 'best.pth'))


if __name__ == '__main__':
    main()
