"""Pemetaan label ChickenAct (15 kelas) -> kelas perilaku sistem.

REVISI PENTING (10 Nov 2026): daftar 15 kelas di whitepaper / dokumen
part 2 TIDAK SAMA dengan isi ChickenAct.zip yang sebenarnya. Yang ada di
zip (diperiksa langsung dari ``ChickenAct.zip``, Zenodo 20672799, md5
terverifikasi — lihat ``ChickenAct_structure.json`` di workspace):

    body_shaking, drinking, dust_bathing, feeding, head_scratching,
    litter_pecking, litter_scratching, pooping, preening, resting,
    running, standing, stretching, walking, wing_flapping

Yang ada di whitepaper tapi TIDAK ADA di data: sitting, pecking_other,
fighting, head_shaking, erratic movement / distress. Artinya: **kelas
AGONISTIC tidak punya satu pun contoh latih di dataset ini** — model
tidak bisa belajar membedakan agresi/distress dari ChickenAct. Lihat
catatan AGONISTIC di bawah.

Pemetaan yang dipakai (berdasarkan fungsi perilaku / etogram ayam
standar — mis. foraging = pecking + scratching di litter; comfort =
preening, stretching, shaking, dust bathing; dan whitepaper §4.3):

    INGESTIVE  (0): feeding (= eating), drinking,
                    litter_pecking (= pecking ground),
                    litter_scratching (foraging: mengais litter cari pakan)
    ACTIVE     (1): walking, running, wing_flapping
    RESTING    (2): resting, standing, preening, dust_bathing,
                    stretching (comfort), head_scratching (grooming),
                    body_shaking (comfort: merapikan bulu)
    AGONISTIC  (3): (DICADANGKAN — tidak ada data latih; lihat di bawah)

    Total terpetakan: 4 + 3 + 7 = 14 kelas (14.990 klip).
    Dikecualikan: pooping (260 klip, 1,7%) — lihat EXCLUDED_CLASSES.

Keputusan AGONISTIC (didokumentasikan, bukan disembunyikan): karena
tidak ada klip fighting / pecking_other / distress di ChickenAct,
classifier Stage 2 HANYA melatih 3 kelas (INGESTIVE/ACTIVE/RESTING).
Id label 3 (AGONISTIC) tetap dicadangkan di taksonomi — kode
(``train.py``) menurunkan ``num_labels`` dari kelas yang benar-benar ada
di manifest, jadi kalau nanti ada data agresi, pipeline yang sama
melatih 4 kelas tanpa perubahan kode. Deteksi agresi/pecking outbreak
di Stage 3 untuk sementara TIDAK bisa mengandalkan output VideoMAE;
harus dari fitur lain (variansi kecepatan, overlap bbox) atau data
tambahan — dicatat di README Stage 2 sebagai pekerjaan terbuka.

Nama whitepaper (eating, pecking_ground, sitting, dsb.) tetap dikenali
sebagai alias supaya pemetaan tidak pecah kalau ada data lain yang
memakai nama itu — tapi di ChickenAct.zip nama-nama itu tidak muncul.
"""

import re

# Taksonomi sistem (urutan = id label; jangan diubah — id 0-2 dipakai
# model 3-kelas saat ini, id 3 dicadangkan untuk AGONISTIC).
CLASS_NAMES_4 = ["INGESTIVE", "ACTIVE", "RESTING", "AGONISTIC"]
CLASS_TO_ID_4 = {name: i for i, name in enumerate(CLASS_NAMES_4)}
ID_TO_CLASS_4 = {i: name for name, i in CLASS_TO_ID_4.items()}

# Kelas yang DIKECUALIKAN dari training (dengan alasan, lihat docstring):
# pooping = peristiwa fisiologis sesaat (buang kotoran), bukan state
# perilaku kesejahteraan — tidak masuk ke satu pun dari 4 kategori.
EXCLUDED_CLASSES = {"pooping"}

# Pemetaan dari nama kelas ChickenAct (sudah dinormalisasi) -> kelas sistem.
# Kunci di sini adalah nama AKTUAL di ChickenAct.zip; nama whitepaper
# yang tidak ada di zip tetap didaftarkan sebagai alias (di bawah).
LABEL15_TO_LABEL4 = {
    # INGESTIVE — konsumsi pakan/air + foraging (mencari makan).
    "feeding": "INGESTIVE",
    "eating": "INGESTIVE",            # alias whitepaper untuk feeding
    "drinking": "INGESTIVE",
    "litter_pecking": "INGESTIVE",    # = pecking ground (whitepaper)
    "pecking_ground": "INGESTIVE",    # alias whitepaper
    "litter_scratching": "INGESTIVE",  # foraging: mengais litter
    # ACTIVE — mobilitas & lokomosi.
    "walking": "ACTIVE",
    "running": "ACTIVE",
    "wing_flapping": "ACTIVE",
    # RESTING — istirahat normal & perawatan tubuh / comfort.
    "resting": "RESTING",
    "sitting": "RESTING",             # alias whitepaper (tidak ada di zip)
    "standing": "RESTING",
    "preening": "RESTING",
    "dust_bathing": "RESTING",
    "stretching": "RESTING",          # comfort: peregangan
    "head_scratching": "RESTING",     # grooming
    "body_shaking": "RESTING",        # comfort: merapikan bulu
    # AGONISTIC — dicadangkan; TIDAK ADA datanya di ChickenAct.
    # Nama-nama ini dikenali supaya tidak error bila muncul di data lain,
    # tapi prepare_chickenact akan mencatat 0 klip untuk kelas ini.
    "pecking_other": "AGONISTIC",
    "fighting": "AGONISTIC",
    "head_shaking": "AGONISTIC",
    "erratic_movement": "AGONISTIC",
    "distress": "AGONISTIC",
    "erratic_movement_distress": "AGONISTIC",
}

# Alias penulisan yang mungkin muncul sebagai nama folder/file.
ALIASES = {
    "dustbathing": "dust_bathing",
    "dust_bath": "dust_bathing",
    "headshaking": "head_shaking",
    "head_shake": "head_shaking",
    "peckingground": "pecking_ground",
    "peckingother": "pecking_other",
    "wingflapping": "wing_flapping",
    "erraticmovement": "erratic_movement",
    "erratic": "erratic_movement",
    "litterpecking": "litter_pecking",
    "litterscratching": "litter_scratching",
    "headscratching": "head_scratching",
    "bodyshaking": "body_shaking",
}


def normalize_class_name(name):
    """Normalisasi nama kelas: kecil, spasi/-/. -> underscore, gabung.

    Contoh: "Head Shaking" -> "head_shaking"; "dustbathing" ->
    "dust_bathing" (lewat ALIASES); "erratic movement / distress" ->
    "erratic_movement_distress".
    """
    n = name.strip().lower()
    n = re.sub(r"[\s\-./]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return ALIASES.get(n, n)


def map_label15(label15):
    """Petakan satu label ChickenAct (15 kelas) ke kelas sistem.

    Mengembalikan nama kelas sistem, atau None kalau label termasuk
    EXCLUDED_CLASSES (pooping). Raise KeyError kalau tidak dikenal —
    jangan ditebak diam-diam; pemanggil harus menangani/mencatatnya.
    """
    key = normalize_class_name(label15)
    if key in EXCLUDED_CLASSES:
        return None
    if key not in LABEL15_TO_LABEL4:
        raise KeyError(
            f"Label ChickenAct tidak dikenal: {label15!r} "
            f"(normalisasi: {key!r}). Perbarui LABEL15_TO_LABEL4 "
            f"setelah memeriksa nama folder sebenarnya di ChickenAct.zip.")
    return LABEL15_TO_LABEL4[key]


def map_label15_to_id(label15):
    """Seperti map_label15, tapi mengembalikan id numerik (atau None)."""
    name = map_label15(label15)
    return None if name is None else CLASS_TO_ID_4[name]


# Nama-nama kelas AKTUAL di ChickenAct.zip (untuk validasi & dokumentasi).
ACTUAL_LABELS_15 = sorted([
    "body_shaking", "drinking", "dust_bathing", "feeding",
    "head_scratching", "litter_pecking", "litter_scratching", "pooping",
    "preening", "resting", "running", "standing", "stretching",
    "walking", "wing_flapping",
])
