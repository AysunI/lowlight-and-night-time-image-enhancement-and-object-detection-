import json
from collections import Counter
from pathlib import Path

GT_JSON = "../../Exdark/Coco_output/exdark_test.json"
DT_JSON = "runs/detect/val2/predictions.json"
OUT_JSON = "runs/detect/val2/prediction_mapped.json"


YOLO_NAMES = ['Bicycle','Boat','Bottle','Bus','Car','Cat',
              'Chair','Cup','Dog','Motorbike','People','Table']

# --- load
gt = json.load(open(GT_JSON))
dt = json.load(open(DT_JSON))

# --- build name -> GT id and YOLO idx -> GT id
name_to_gtid = {c["name"]: c["id"] for c in gt["categories"]}   # 1..12
cls_to_catid = {i: name_to_gtid[name] for i, name in enumerate(YOLO_NAMES)}  # 0..11 → 1..12
valid_gt_ids = set(name_to_gtid.values())

# --- quick diagnostics on incoming DT category_id distribution
raw_ids = Counter(d["category_id"] for d in dt)
print("DT category_id histogram (raw):", dict(raw_ids))

# --- remap image_id strings -> ints
stem2id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}

fixed = []
bad_img, bad_cat = 0, 0
alien_ids = Counter()

for d in dt:
    # image_id
    img = d["image_id"]
    if isinstance(img, str):
        nid = stem2id.get(Path(img).stem)
        if nid is None:
            bad_img += 1
            continue
        d["image_id"] = nid
    elif not isinstance(img, int):
        bad_img += 1
        continue

    # category_id
    cid = d["category_id"]
    if cid in valid_gt_ids:
        # already 1..12 — keep as is
        pass
    elif cid in cls_to_catid:
        # 0..11 → map to 1..12
        d["category_id"] = cls_to_catid[cid]
    else:
        # something else (e.g., 13, 27, 58...)
        bad_cat += 1
        alien_ids[cid] += 1
        continue

    fixed.append(d)

print(f"Kept {len(fixed)} / {len(dt)} detections "
      f"(dropped {bad_img} bad image_id, {bad_cat} bad category_id).")
if alien_ids:
    print("Alien class IDs found (not in your 12 classes):", dict(alien_ids))

json.dump(fixed, open(OUT_JSON, "w"))
print("Wrote:", OUT_JSON)
