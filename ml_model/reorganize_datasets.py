import os
import sys
import json
import glob
import shutil
import csv
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
MANIFEST_PATH = os.path.join(SCRIPT_DIR, 'dataset_reorganize_manifest.json')

HAM_DIR = os.path.join(DATASET_DIR, 'HAM10000')
HAM_META = os.path.join(HAM_DIR, 'HAM10000_metadata.csv')
HAM_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
HAM_IMAGE_FOLDERS = ['HAM10000_images_part_1', 'HAM10000_images_part_2',
                     'HAM10000_images', 'images']

SC_DIR = os.path.join(DATASET_DIR, 'Skin_Cancer')
SC_META = os.path.join(SC_DIR, 'metadata.csv')
SC_CLASSES = ['BCC', 'ACK', 'NEV', 'SEK', 'SCC', 'MEL']
SC_IMAGE_FOLDERS = ['imgs_part_1', 'imgs_part_2', 'imgs_part_3']

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')


def find_images_by_stem(dirs):
    """Map filename-stem -> full path over a list of directories (non-recursive)."""
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(IMG_EXTS):
                stem = os.path.splitext(f)[0]
                out[stem] = os.path.join(d, f)
    return out


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plan_ham10000():
    rows = read_csv(HAM_META)
    images = find_images_by_stem([os.path.join(HAM_DIR, fd) for fd in HAM_IMAGE_FOLDERS])
    moves = []          # (src, dst)
    missing = []        # image_id in metadata but not on disk
    for row in rows:
        dx = row.get('dx', '').strip().lower()
        img_id = row.get('image_id', '').strip()
        if dx not in HAM_CLASSES:
            continue
        src = images.get(img_id)
        if src is None:
            missing.append(img_id)
            continue
        dst_dir = os.path.join(HAM_DIR, dx)
        dst = os.path.join(dst_dir, os.path.basename(src))
        moves.append((src, dst))
    # on-disk files not referenced by metadata
    meta_ids = {r.get('image_id', '').strip() for r in rows}
    unmapped = sorted(set(images.keys()) - meta_ids)
    return moves, missing, unmapped


def plan_skin_cancer():
    rows = read_csv(SC_META)
    images = find_images_by_stem([os.path.join(SC_DIR, fd) for fd in SC_IMAGE_FOLDERS])
    moves, missing = [], []
    for row in rows:
        diag = row.get('diagnostic', '').strip().upper()
        img_id = row.get('img_id', '').strip()
        if diag not in SC_CLASSES:
            continue
        stem = os.path.splitext(img_id)[0]
        src = images.get(stem)
        if src is None:
            missing.append(img_id)
            continue
        dst_dir = os.path.join(SC_DIR, diag)
        dst = os.path.join(dst_dir, os.path.basename(src))
        moves.append((src, dst))
    meta_ids = {os.path.splitext(r.get('img_id', '').strip())[0] for r in rows}
    unmapped = sorted(set(images.keys()) - meta_ids)
    return moves, missing, unmapped


def report_ptbxl():
    """Map each PTB-XL record to its primary diagnostic superclass (read-only)."""
    sys.path.insert(0, SCRIPT_DIR)
    # Import the mapping logic from the training script without running it.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'thi', os.path.join(SCRIPT_DIR, 'train_heart_image_model.py'))
    # The module prints a lot; import the pieces we need manually instead.
    SCP_TO_CLASS = {
        'NORM': 0, 'SR': 0,
        'IMI': 1, 'AMI': 1, 'LMI': 1, 'PMI': 1, 'ASMI': 1, 'ILMI': 1, 'IPLMI': 1,
        'IPMI': 1, 'MI': 1,
        'VPRE': 2, 'WPW': 2, 'STTC': 2, 'NST_': 2,
        'CD': 3, 'IVCB': 3,
        'RAO': 4, 'RAE': 4, 'SEHYP': 4, 'HYP': 4,
    }
    NAMES = {0: 'Normal', 1: 'MI', 2: 'Arrhythmia', 3: 'Heart Failure', 4: 'Hypertrophy'}

    def assign(scp_codes):
        if not isinstance(scp_codes, dict):
            scp_codes = {}
        best_norm = -1.0
        found = {}
        for code, lik in scp_codes.items():
            cu = str(code).upper()
            l_val = float(lik)
            if cu in ('NORM', 'SR'):
                if l_val > best_norm:
                    best_norm = l_val
                continue
            if cu in SCP_TO_CLASS:
                cls = SCP_TO_CLASS[cu]
                found[cls] = max(found.get(cls, 0.0), l_val)
        if found:
            return max(found, key=found.get)
        return 0 if best_norm >= 0 else None

    db = os.path.join(os.path.join(DATASET_DIR, 'ptb-xl'), 'ptbxl_database.csv')
    if not os.path.exists(db):
        print("ptbxl_database.csv not found; skipping PTB-XL report.")
        return
    rows = read_csv(db)
    dist = Counter()
    skipped = 0
    for row in rows:
        scp_raw = row.get('scp_codes', '')
        try:
            import ast
            scp = ast.literal_eval(scp_raw) if scp_raw else {}
        except Exception:
            scp = {}
        cls = assign(scp)
        if cls is None:
            skipped += 1
            continue
        dist[NAMES[cls]] += 1
    print(f"\nPTB-XL primary-diagnostic superclass distribution "
          f"({len(rows)} records, {skipped} unclassifiable):")
    for name, cnt in dist.most_common():
        print(f"  {name:15s} {cnt}")


def apply(moves):
    manifest = []
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        manifest.append({'source': src, 'destination': dst})
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Moved {len(moves)} files. Manifest -> {MANIFEST_PATH}")


def main():
    args = sys.argv[1:]
    do_apply = '--apply' in args
    do_report = '--report-ptbxl' in args

    if do_report:
        report_ptbxl()
        return

    print("=" * 70)
    print("HAM10000")
    print("=" * 70)
    ham_moves, ham_missing, ham_unmapped = plan_ham10000()
    ham_by_class = Counter(os.path.basename(os.path.dirname(d)) for _, d in ham_moves)
    for c in HAM_CLASSES:
        print(f"  {c:8s} {ham_by_class.get(c, 0)}")
    print(f"  TOTAL moves: {len(ham_moves)}  missing-on-disk: {len(ham_missing)}  "
          f"on-disk-but-unmapped: {len(ham_unmapped)}")
    if ham_missing:
        print(f"  missing sample: {ham_missing[:5]}")
    if ham_unmapped:
        print(f"  unmapped sample: {ham_unmapped[:5]}")

    print()
    print("=" * 70)
    print("Skin_Cancer")
    print("=" * 70)
    sc_moves, sc_missing, sc_unmapped = plan_skin_cancer()
    sc_by_class = Counter(os.path.basename(os.path.dirname(d)) for _, d in sc_moves)
    for c in SC_CLASSES:
        print(f"  {c:6s} {sc_by_class.get(c, 0)}")
    print(f"  TOTAL moves: {len(sc_moves)}  missing-on-disk: {len(sc_missing)}  "
          f"on-disk-but-unmapped: {len(sc_unmapped)}")
    if sc_missing:
        print(f"  missing sample: {sc_missing[:5]}")
    if sc_unmapped:
        print(f"  unmapped sample: {sc_unmapped[:5]}")

    if do_apply:
        print("\n[APPLY] Moving files...")
        apply(ham_moves + sc_moves)
        print("[DONE]")
    else:
        print("\nDry-run only. Re-run with --apply to move files.")


if __name__ == '__main__':
    main()
