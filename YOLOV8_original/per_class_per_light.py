import os
import json
import csv
import glob
import yaml
from collections import defaultdict, Counter
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATA_YAML   = "data.yaml" 
GT_JSON     = "../../Exdark/Coco_output/exdark_test.json"
DT_JSON     = "runs/detect/val2/prediction_mapped.json"  
EXDARK_IMAGECLASS = "../../imageclasslist.txt"            
OUT_DIR     = "runs/metrics_out_original"

EXDARK_LIGHT_NAMES = {
    1:"Low", 2:"Ambient", 3:"Object", 4:"Single", 5:"Weak",
    6:"Strong", 7:"Screen", 8:"Window", 9:"Shadow", 10:"Twilight"
}

os.makedirs(OUT_DIR, exist_ok=True)
def load_names_from_yaml(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"DATA_YAML not found: {p}")
    with open(p, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names", None)
    if names is None:
        raise ValueError(f"`names` not found in {p}")
    if isinstance(names, dict):
        # ensure 0..N-1 order
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError(f"`names` malformed in {p}: {type(names)}")
    return list(names)

def load_exdark_imageclass(path):
    """
    Parse ExDark imageclass.txt with columns:
    Name | Class | Light | In/Out | Train/Val/Test
    Returns: {basename -> lighting_name}
    """
    out = {}
    with open(path, "r") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if "name" in low and "light" in low:
                # header
                continue
            parts = line.replace("|", " ").split()
            if len(parts) < 3:
                continue
            fn = os.path.basename(parts[0])
            light_token = parts[2]
            if not light_token.isdigit():
                continue
            lid = int(light_token)
            out[fn] = EXDARK_LIGHT_NAMES.get(lid, "Unknown")
    return out

def eval_slice(cocoGt, cocoDt, img_ids=None, cat_ids=None, quiet=True):
    ev = COCOeval(cocoGt, cocoDt, iouType="bbox")
    if img_ids is not None:
        ev.params.imgIds = img_ids
    if cat_ids is not None:
        ev.params.catIds = cat_ids
    ev.evaluate()
    ev.accumulate()
    # COCOeval.stats is filled by summarize(); silence if quiet
    if quiet:
        import io, sys
        _stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            ev.summarize()
        finally:
            sys.stdout = _stdout
    else:
        ev.summarize()
    s = ev.stats if getattr(ev, "stats", None) is not None and len(ev.stats) == 12 else [float("nan")] * 12
    return dict(
        AP=float(s[0]), AP50=float(s[1]), AP75=float(s[2]),
        APs=float(s[3]), APm=float(s[4]), APl=float(s[5]),
        AR1=float(s[6]), AR10=float(s[7]), AR100=float(s[8]),
        ARs=float(s[9]), ARm=float(s[10]), ARl=float(s[11]),
    )

def prepare_predictions(cocoGt, dt, yolo_names):
    """
    - map image_id strings -> numeric GT ids (by stem)
    - map YOLO class idx (0..N-1) -> GT category_id via names
    - keep entries that match GT image & category sets
    Returns fixed list and diagnostics.
    """
    stem2id = {Path(im["file_name"]).stem: im["id"] for im in cocoGt.dataset["images"]}
    name_to_gtid = {c["name"]: c["id"] for c in cocoGt.dataset["categories"]}
    # sanity: warn if names missing in GT
    missing = [nm for nm in yolo_names if nm not in name_to_gtid]
    if missing:
        raise ValueError(f"These YOLO class names are not present in GT categories: {missing}")

    cls_to_gtid = {i: name_to_gtid[nm] for i, nm in enumerate(yolo_names)}

    valid_gt_img_ids = {im["id"] for im in cocoGt.dataset["images"]}
    valid_gt_cat_ids = set(name_to_gtid.values())

    # diagnostics
    raw_cids = Counter(d.get("category_id") for d in dt)
    print("Raw prediction category_id histogram (top 10):", dict(raw_cids.most_common(10)))

    fixed = []
    bad_img = bad_cat = 0
    alien_ids = Counter()

    for d in dt:
        # image_id mapping
        img = d.get("image_id")
        if isinstance(img, str):
            nid = stem2id.get(Path(img).stem)
            if nid is None:
                bad_img += 1
                continue
            d["image_id"] = nid
        elif not isinstance(img, int):
            bad_img += 1
            continue

        # category_id mapping
        cid = d.get("category_id")
        if cid in valid_gt_cat_ids:
            pass  # already GT ids (1..K)
        elif cid in cls_to_gtid:
            d["category_id"] = cls_to_gtid[cid]
        else:
            bad_cat += 1
            alien_ids[cid] += 1
            continue

        if d["image_id"] in valid_gt_img_ids and d["category_id"] in valid_gt_cat_ids:
            fixed.append(d)

    diag = {
        "dropped_bad_image_id": bad_img,
        "dropped_bad_category_id": bad_cat,
        "alien_category_ids": dict(alien_ids),
        "kept": len(fixed),
        "total": len(dt),
    }
    return fixed, diag

def main():
    # 1) Load GT via COCO and patch missing top-level keys
    cocoGt = COCO(GT_JSON)
    cocoGt.dataset.setdefault("info", {})
    cocoGt.dataset.setdefault("licenses", [])

    # 2) Resolve predictions path
    pred_path = DT_JSON
    if pred_path is None:
        candidates = glob.glob('runs/detect/**/predictions.json', recursive=True)
        if not candidates:
            raise FileNotFoundError("No predictions.json found under runs/detect/**/. "
                                    "Run YOLO val with save_json=True first.")
        pred_path = max(candidates, key=os.path.getmtime)
    print("Using detections:", pred_path)

    with open(pred_path, "r") as f:
        preds_raw = json.load(f)

    # 3) Load YOLO names strictly from YAML (no fallback)
    yolo_names = load_names_from_yaml(DATA_YAML)
    print("YOLO names:", yolo_names)

    # 4) Map predictions to GT ids/categories
    preds_fixed, diag = prepare_predictions(cocoGt, preds_raw, yolo_names)
    print(f"Kept {diag['kept']} / {diag['total']} detections "
          f"(dropped {diag['dropped_bad_image_id']} bad image_id, "
          f"{diag['dropped_bad_category_id']} bad category_id).")
    if diag["alien_category_ids"]:
        print("Alien (unexpected) category IDs found:", diag["alien_category_ids"])
    if not preds_fixed:
        raise RuntimeError("No predictions left after mapping/filtering. Check class names and GT IDs.")

    # 5) Build cocoDt
    cocoDt = cocoGt.loadRes(preds_fixed)

    imgs = cocoGt.loadImgs(cocoGt.getImgIds())
    cats = sorted(cocoGt.loadCats(cocoGt.getCatIds()), key=lambda c: c["id"])
    id2name = {c["id"]: c["name"] for c in cats}

    # 6) Per-class CSV
    perclass_rows = []
    for cid in sorted(id2name.keys()):
        cname = id2name[cid]
        m = eval_slice(cocoGt, cocoDt, cat_ids=[cid], quiet=True)
        perclass_rows.append(dict(class_id=cid, class_name=cname, **m))

    perclass_csv = os.path.join(OUT_DIR, "per_class_metrics.csv")
    with open(perclass_csv, "w", newline="") as f:
        fields = ["class_id","class_name","AP","AP50","AP75","APs","APm","APl",
                  "AR1","AR10","AR100","ARs","ARm","ARl"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in perclass_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("Saved:", perclass_csv)

    # 7) Per-lighting overall CSV
    light_map = load_exdark_imageclass(EXDARK_IMAGECLASS)
    id2file = {im["id"]: os.path.basename(im["file_name"]) for im in imgs}
    cond_to_imgids = defaultdict(list)
    for im in imgs:
        cond = light_map.get(id2file[im["id"]], "Unknown")
        cond_to_imgids[cond].append(im["id"])

    perlight_rows = []
    for cond, ids in sorted(cond_to_imgids.items()):
        m = eval_slice(cocoGt, cocoDt, img_ids=ids, quiet=True)
        perlight_rows.append(dict(condition=cond, count_images=len(ids), **m))

    perlight_csv = os.path.join(OUT_DIR, "per_lighting_overall.csv")
    with open(perlight_csv, "w", newline="") as f:
        fields = ["condition","count_images","AP","AP50","AP75","APs","APm","APl",
                  "AR1","AR10","AR100","ARs","ARm","ARl"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in perlight_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("Saved:", perlight_csv)

    # 8) Per-class within each lighting CSV
    perlightclass_csv = os.path.join(OUT_DIR, "per_lighting_per_class.csv")
    rows = []
    for cond, ids in sorted(cond_to_imgids.items()):
        for cid in sorted(id2name.keys()):
            cname = id2name[cid]
            m = eval_slice(cocoGt, cocoDt, img_ids=ids, cat_ids=[cid], quiet=True)
            rows.append(dict(condition=cond, class_id=cid, class_name=cname, **m))
    with open(perlightclass_csv, "w", newline="") as f:
        fields = ["condition","class_id","class_name","AP","AP50","AP75","APs","APm","APl",
                  "AR1","AR10","AR100","ARs","ARm","ARl"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("Saved:", perlightclass_csv)

    # 9) Quick sanity stats
    gt_ids = {im["id"] for im in imgs}
    dt_ids = {d["image_id"] for d in preds_fixed}
    print(f"Unique GT images: {len(gt_ids)} | images with ≥1 prediction: {len(dt_ids)} | overlap: {len(gt_ids & dt_ids)}")

if __name__ == "__main__":
    main()
