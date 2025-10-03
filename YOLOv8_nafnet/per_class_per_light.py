# eval_yolo_summary_fixed.py
import os, json, csv, yaml
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATA_YAML   = "data.yaml"  
GT_JSON     = "../../Exdark/resized_coco_json/instances_test.json"
DT_JSON     = "runs/detect/exdark_test_eval/prediction_mapped.json"  
EXDARK_IMAGECLASS = "../../imageclasslist.txt"
OUT_DIR     = "runs/metrics_out"
# ----------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

EXDARK_LIGHT_NAMES = {
    1:"Low",2:"Ambient",3:"Object",4:"Single",5:"Weak",
    6:"Strong",7:"Screen",8:"Window",9:"Shadow",10:"Twilight"
}

def load_names_from_yaml(p):
    if not os.path.exists(p): return None
    with open(p, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names

def load_exdark_imageclass(path):
    """
    Parse ExDark imageclass.txt with columns:
    Name | Class | Light | In/Out | Train/Val/Test
    Returns: {filename -> lighting_name}
    """
    out = {}
    with open(path, "r") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if "name" in low and "light" in low:
                continue

            # normalize separators: turn pipes into spaces then split
            parts = line.replace("|", " ").split()
            if len(parts) < 3:
                # malformed row; skip
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

    # run eval
    ev.evaluate()
    ev.accumulate()
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

    s = ev.stats if getattr(ev, "stats", None) is not None and len(ev.stats) == 12 else [float("nan")]*12
    return dict(
        AP=float(s[0]), AP50=float(s[1]), AP75=float(s[2]),
        APs=float(s[3]), APm=float(s[4]), APl=float(s[5]),
        AR1=float(s[6]), AR10=float(s[7]), AR100=float(s[8]),
        ARs=float(s[9]), ARm=float(s[10]), ARl=float(s[11]),
    )

def main():
    # Load GT and predictions
    cocoGt = COCO(GT_JSON)
    with open(DT_JSON, "r") as f:
        preds = json.load(f)

    # Filter predictions to valid GT image/category ids (defensive & fair)
    gt_img_ids = {im["id"] for im in cocoGt.dataset["images"]}
    gt_cat_ids = {c["id"] for c in cocoGt.dataset["categories"]}
    preds = [d for d in preds if d.get("image_id") in gt_img_ids and d.get("category_id") in gt_cat_ids]
    cocoDt = cocoGt.loadRes(preds)

    imgs = cocoGt.loadImgs(cocoGt.getImgIds())
    cats = sorted(cocoGt.loadCats(cocoGt.getCatIds()), key=lambda c: c["id"])
    id2name = {c["id"]: c["name"] for c in cats}

    # sanity print
    names_yaml = load_names_from_yaml(DATA_YAML)
    if names_yaml:
        print("YOLO names (from yaml):", names_yaml)
    print("GT class order:", [id2name[c["id"]] for c in cats])

    # ----- Per-class CSV -----
    perclass_rows = []
    for cid, cname in id2name.items():
        m = eval_slice(cocoGt, cocoDt, cat_ids=[cid])
        perclass_rows.append(dict(class_id=cid, class_name=cname, **m))

    perclass_csv = os.path.join(OUT_DIR, "per_class_metrics.csv")
    with open(perclass_csv, "w", newline="") as f:
        fields = ["class_id","class_name","AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in perclass_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("Saved:", perclass_csv)

    # ----- Per-lighting CSV -----
    light_map = load_exdark_imageclass(EXDARK_IMAGECLASS)  # filename -> condition
    id2file = {im["id"]: os.path.basename(im["file_name"]) for im in imgs}
    cond_to_imgids = defaultdict(list)
    for im in imgs:
        cond = light_map.get(id2file[im["id"]], "Unknown")
        cond_to_imgids[cond].append(im["id"])

    perlight_rows = []
    for cond, ids in sorted(cond_to_imgids.items()):
        m = eval_slice(cocoGt, cocoDt, img_ids=ids)
        perlight_rows.append(dict(condition=cond, count_images=len(ids), **m))

    perlight_csv = os.path.join(OUT_DIR, "per_lighting_overall.csv")
    with open(perlight_csv, "w", newline="") as f:
        fields = ["condition","count_images","AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in perlight_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("Saved:", perlight_csv)

    # ----- per-class within each lighting -----
    perlightclass_csv = os.path.join(OUT_DIR, "per_lighting_per_class.csv")
    rows = []
    for cond, ids in sorted(cond_to_imgids.items()):
        for cid, cname in id2name.items():
            m = eval_slice(cocoGt, cocoDt, img_ids=ids, cat_ids=[cid])
            rows.append(dict(condition=cond, class_id=cid, class_name=cname, **m))
    with open(perlightclass_csv, "w", newline="") as f:
        fields = ["condition","class_id","class_name","AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("Saved:", perlightclass_csv)

if __name__ == "__main__":
    main()
