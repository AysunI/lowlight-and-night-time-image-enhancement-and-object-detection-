# eval_exdark_with_lighting_and_viz.py
# Post-training evaluation for ExDark (torchvision Faster R-CNN).
# Outputs under OUT_DIR:
#  - coco_<split>_detections.json
#  - overall_metrics_<split>.txt
#  - per_class_ap_<split>.csv
#  - lighting_stats_<split>.csv
#  - per_class_by_lighting_<split>.csv
#  - pr_curves/<class>_<split>.csv
#  - viz/*.jpg  (TP/FP/FN visualizations)

import os
import io
import json
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMAGE_ROOT   = "../../Exdark/zero-dce++"
ANN_JSON     = "../../Exdark/Coco_output/exdark_test.json"         
CKPT         = "../../Exdark/exdark_frcnn_outputs_zero-dce/model_best.pth"  
SPLIT_TABLE  = "../../imageclasslist.txt"                          
OUT_DIR      = "../../Exdark/exdark_frcnn_outputs_zero-dce/test_result"
SPLIT_NAME   = "test"                    
BATCH_SIZE   = 2
NUM_WORKERS  = 4
SCORE_THRESH_SAVE = 0.0                  # save all detections; raise if files too large
# visualization settings 
VIZ_NUM_IMAGES   = 12                    # number of examples to save
VIZ_SCORE_THR    = 0.35                  # prediction threshold for drawing
VIZ_IOU_THR      = 0.5                   # IoU for TP matching
RANDOM_SEED      = 123                   # set None for different picks each run

LIGHTING_NAMES = {
    1:"Low", 2:"Ambient", 3:"Object", 4:"Single", 5:"Weak",
    6:"Strong", 7:"Screen", 8:"Window", 9:"Shadow", 10:"Twilight"}

def collate(batch): return tuple(zip(*batch))

# ---------------- Dataset & model ----------------
class CocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, img_root):
        self.coco = COCO(ann_file)
        self.img_root = img_root
        self.ids = list(sorted(self.coco.imgs.keys()))
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats_sorted = sorted(cats, key=lambda c: c["id"])
        self.catid2contig = {c["id"]: i+1 for i, c in enumerate(cats_sorted)}
        self.contig2catid = {i+1: c["id"] for i, c in enumerate(cats_sorted)}
        self.contig2name  = {i+1: c["name"] for i, c in enumerate(cats_sorted)}
        self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_root, info["file_name"])).convert("RGB")
        return self.tf(img), {"image_id": torch.tensor([img_id], dtype=torch.int64)}

def load_model(num_classes, ckpt_path, device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, 1+num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model

# ---------------- Split table (lighting) ----------------
def read_lighting_map(split_table_path):
    """Return dict: basename -> lighting_id (1..10)."""
    m = {}
    with open(split_table_path, "r", errors="ignore") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line: continue
            if first and line.lower().startswith("name"):
                first = False; 
                continue
            parts = line.split()  # whitespace-separated
            if len(parts) < 5: continue
            base = os.path.basename(parts[0])
            try:
                light_id = int(parts[2])
            except:
                continue
            m[base] = light_id
    return m

# ---------------- Inference & COCO eval ----------------
@torch.no_grad()
def run_detections(model, dataset, loader, device, out_dir, split_name, score_thresh=0.0):
    """Run detector and save standard COCO detection json."""
    results = []
    for imgs, targets in tqdm(loader, desc=f"Inference ({split_name})"):
        imgs = [img.to(device) for img in imgs]
        outs = model(imgs)
        for out, t in zip(outs, targets):
            img_id = int(t["image_id"].item())
            boxes  = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            keep = scores >= float(score_thresh)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            for (x1,y1,x2,y2), s, l in zip(boxes, scores, labels):
                results.append({
                    "image_id": img_id,
                    "category_id": int(dataset.contig2catid[int(l)]),
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score": float(s)})
    os.makedirs(out_dir, exist_ok=True)
    det_file = os.path.join(out_dir, f"coco_{split_name}_detections.json")
    json.dump(results, open(det_file, "w"))
    return det_file

def coco_eval_and_capture(coco_gt, coco_dt, out_dir, split_name, img_ids=None):
    """Run COCOeval (optionally on subset). Return (coco_eval, text_summary)."""
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids is not None:
        ev.params.imgIds = img_ids
    with io.StringIO() as buf, redirect_stdout(buf):
        ev.evaluate(); ev.accumulate(); ev.summarize()
        text = buf.getvalue()
    with open(os.path.join(out_dir, f"overall_metrics_{split_name}.txt"), "w") as f:
        f.write(text)
    return ev, text

def save_per_class_metrics(coco_eval, dataset, out_dir, split_name):
    """Save per-class AP, AP50, AP75, mean precision, mean recall to CSV."""
    P = coco_eval.eval.get("precision", None)  # [T, R, K, A, M]
    R = coco_eval.eval.get("recall", None)     # [T, K, A, M]
    if P is None or R is None:
        print("Warning: precision/recall not available in coco_eval.")
        return None

    K = P.shape[2]
    cat_ids = dataset.coco.getCatIds()
    cats = dataset.coco.loadCats(cat_ids)
    names = [c["name"] for c in sorted(cats, key=lambda c: c["id"])]

    rows = [["class","AP","AP50","AP75","meanPrecision","meanRecall"]]
    for k in range(K):
        p_all = P[:, :, k, 0, -1]; p_all = p_all[p_all > -1]
        ap = float(p_all.mean()) if p_all.size else float("nan")
        p50 = P[0, :, k, 0, -1]; p50 = p50[p50 > -1]
        ap50 = float(p50.mean()) if p50.size else float("nan")
        p75 = P[5, :, k, 0, -1]; p75 = p75[p75 > -1]
        ap75 = float(p75.mean()) if p75.size else float("nan")
        mean_prec = float(p_all.mean()) if p_all.size else float("nan")
        r_all = R[:, k, 0, -1]; r_all = r_all[r_all > -1]
        mean_rec = float(r_all.mean()) if r_all.size else float("nan")
        rows.append([names[k], f"{ap:.4f}", f"{ap50:.4f}", f"{ap75:.4f}", f"{mean_prec:.4f}", f"{mean_rec:.4f}"])

    csv_path = os.path.join(out_dir, f"per_class_ap_{split_name}.csv")
    with open(csv_path, "w") as f:
        for r in rows: f.write(",".join(map(str, r)) + "\n")
    print(f"Per-class metrics saved to: {csv_path}")
    return {"precision": P, "recall": R, "rec_thrs": coco_eval.params.recThrs, "class_names": names}

def save_pr_curves(perclass_data, out_dir, split_name):
    """Save PR curves per class at IoU=0.50 and IoU=0.75 as CSV."""
    if perclass_data is None: return
    P = perclass_data["precision"]; rec_thrs = perclass_data["rec_thrs"]; names = perclass_data["class_names"]
    os.makedirs(os.path.join(out_dir, "pr_curves"), exist_ok=True)
    for k, name in enumerate(names):
        p50 = P[0, :, k, 0, -1]; m50 = p50 > -1
        p75 = P[5, :, k, 0, -1]; m75 = p75 > -1
        rows = [["recall","precision_iou50","precision_iou75"]]
        for i in range(len(rec_thrs)):
            r = rec_thrs[i]
            pr50 = f"{float(p50[i]):.4f}" if m50[i] else ""
            pr75 = f"{float(p75[i]):.4f}" if m75[i] else ""
            rows.append([f"{float(r):.4f}", pr50, pr75])
        csv_path = os.path.join(out_dir, "pr_curves", f"{name}_{split_name}.csv")
        with open(csv_path, "w") as f:
            for r in rows: f.write(",".join(r) + "\n")
    print(f"Per-class PR curves saved under: {os.path.join(out_dir,'pr_curves')}")

def eval_per_lighting(coco_gt, coco_dt, dataset, split_table_path, out_dir, split_name):
    """Overall metrics per lighting and class×lighting AP matrix."""
    light_map = read_lighting_map(split_table_path)  # base -> lid
    fn2imgid = {os.path.basename(info["file_name"]): iid for iid, info in coco_gt.imgs.items()}

    lighting_to_imgIds = {lid: [] for lid in LIGHTING_NAMES.keys()}
    for base, lid in light_map.items():
        iid = fn2imgid.get(base)
        if iid is not None and lid in lighting_to_imgIds:
            lighting_to_imgIds[lid].append(iid)

    # Overall per lighting
    rows = [["Lighting","AP","AP50","AP75","AR@1","AR@10","AR@100","num_images"]]
    # Class × lighting matrix
    cat_ids = dataset.coco.getCatIds()
    cats = dataset.coco.loadCats(cat_ids)
    class_names = [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
    class_matrix = {name: {} for name in class_names}

    for lid, img_ids in lighting_to_imgIds.items():
        if not img_ids: 
            continue
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = img_ids
        with io.StringIO() as buf, redirect_stdout(buf):
            ev.evaluate(); ev.accumulate(); ev.summarize()
        stats = ev.stats
        ap, ap50, ap75 = (float(stats[0]), float(stats[1]), float(stats[2])) if stats is not None else (0.0,0.0,0.0)
        ar1, ar10, ar100 = (float(stats[6]), float(stats[7]), float(stats[8])) if stats is not None else (0.0,0.0,0.0)
        rows.append([LIGHTING_NAMES.get(lid,str(lid)), f"{ap:.4f}", f"{ap50:.4f}", f"{ap75:.4f}", f"{ar1:.4f}", f"{ar10:.4f}", f"{ar100:.4f}", str(len(img_ids))])

        # per-class AP within this lighting
        P = ev.eval.get("precision", None)
        if P is not None:
            K = P.shape[2]
            for k in range(K):
                p_all = P[:, :, k, 0, -1]; p_all = p_all[p_all > -1]
                ap_k = float(p_all.mean()) if p_all.size else float("nan")
                cname = class_names[k]
                class_matrix[cname][LIGHTING_NAMES.get(lid,str(lid))] = f"{ap_k:.4f}"

    # Save per-lighting overall
    csv1 = os.path.join(out_dir, f"lighting_stats_{split_name}.csv")
    with open(csv1, "w") as f:
        for r in rows: f.write(",".join(map(str, r)) + "\n")
    print(f"Per-lighting metrics saved to: {csv1}")

    # Save class × lighting matrix
    lighting_cols = [LIGHTING_NAMES[k] for k in sorted(LIGHTING_NAMES.keys())]
    header = ["class"] + lighting_cols
    lines = [header]
    for cname in class_names:
        row = [cname] + [class_matrix[cname].get(LIGHTING_NAMES[k], "") for k in sorted(LIGHTING_NAMES.keys())]
        lines.append(row)
    csv2 = os.path.join(out_dir, f"per_class_by_lighting_{split_name}.csv")
    with open(csv2, "w") as f:
        for r in lines: f.write(",".join(r) + "\n")
    print(f"Per-class by lighting AP matrix saved to: {csv2}")

# ---------------- Visualization (TP/FP/FN) ----------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    ua = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    ub = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    union = ua + ub - inter
    return inter/union if union>0 else 0.0

def match_tp_fp_fn(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr=0.5):
    used_gt = set(); tp, fp = [], []
    order = sorted(range(len(pred_boxes)), key=lambda i: float(pred_scores[i]), reverse=True)
    for i in order:
        pb, pl = pred_boxes[i], int(pred_labels[i])
        best_j, best_iou = -1, 0.0
        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if j in used_gt or int(gl) != pl: 
                continue
            iou = iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp.append((i, best_j)); used_gt.add(best_j)
        else:
            fp.append(i)
    fn = [j for j in range(len(gt_boxes)) if j not in used_gt]
    return tp, fp, fn

def draw_box(draw, box, color, width=3):
    x1,y1,x2,y2 = map(int, box)
    for w in range(width):
        draw.rectangle((x1-w, y1-w, x2+w, y2+w), outline=color)

def put_text(draw, xy, txt, fill=(255,255,255)):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = None
    draw.text(xy, txt, fill=fill, font=font)

@torch.no_grad()
def visualize_tp_fp_fn(model, dataset, device, out_dir, split_name, num_images=12, score_thr=0.35, iou_thr=0.5, seed=123):
    os.makedirs(out_dir, exist_ok=True)
    ids = list(dataset.coco.imgs.keys())
    if seed is not None:
        random.seed(seed)
        random.shuffle(ids)
    else:
        random.shuffle(ids)
    picked = ids[:num_images]

    for img_id in tqdm(picked, desc="Visualizing TP/FP/FN"):
        info = dataset.coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_ROOT, info["file_name"])
        img_pil = Image.open(img_path).convert("RGB")
        tensor = transforms.ToTensor()(img_pil).to(device)

        pred = model([tensor])[0]
        boxes_p = pred["boxes"].detach().cpu().tolist()
        labels_p = pred["labels"].detach().cpu().tolist()
        scores_p = pred["scores"].detach().cpu().tolist()

        keep = [i for i,s in enumerate(scores_p) if s >= score_thr]
        boxes_p = [boxes_p[i] for i in keep]
        labels_p = [labels_p[i] for i in keep]
        scores_p = [scores_p[i] for i in keep]

        # GT
        ann_ids = dataset.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = dataset.coco.loadAnns(ann_ids)
        boxes_g, labels_g = [], []
        for a in anns:
            x,y,w,h = a["bbox"]
            boxes_g.append([x,y,x+w,y+h])
            labels_g.append(dataset.catid2contig[a["category_id"]])

        tp, fp, fn = match_tp_fp_fn(boxes_p, labels_p, scores_p, boxes_g, labels_g, iou_thr=iou_thr)

        img = img_pil.copy()
        draw = ImageDraw.Draw(img)
        # faint GT for context
        for gb in boxes_g:
            draw_box(draw, gb, color=(180,180,180), width=1)
        # TP green
        for i_pred, _ in tp:
            b = boxes_p[i_pred]; l = int(labels_p[i_pred]); s = scores_p[i_pred]
            cname = dataset.contig2name[l]
            draw_box(draw, b, color=(0,255,0), width=3)
            x1,y1,_,_ = map(int, b)
            put_text(draw, (x1, max(0, y1-18)), f"{cname} {s:.2f}", fill=(0,255,0))
        # FP red
        for i_pred in fp:
            b = boxes_p[i_pred]; l = int(labels_p[i_pred]); s = scores_p[i_pred]
            cname = dataset.contig2name[l]
            draw_box(draw, b, color=(255,0,0), width=3)
            x1,y1,_,_ = map(int, b)
            put_text(draw, (x1, max(0, y1-18)), f"{cname} {s:.2f} (FP)", fill=(255,0,0))
        # FN yellow
        for j_gt in fn:
            b = boxes_g[j_gt]; l = int(labels_g[j_gt]); cname = dataset.contig2name[l]
            draw_box(draw, b, color=(255,215,0), width=3)
            x1,y1,_,_ = map(int, b)
            put_text(draw, (x1, max(0, y1-18)), f"{cname} (FN)", fill=(255,215,0))

        put_text(draw, (10,10), f"TP=green, FP=red, FN=yellow | thr={score_thr}, IoU={iou_thr}", fill=(255,255,255))

        save_name = os.path.basename(info["file_name"]).replace("/", "_")
        save_path = os.path.join(out_dir, f"viz_{split_name}_{save_name}")
        img.save(save_path)

    print(f"Saved {len(picked)} visualizations to: {out_dir}")

# ---------------- Main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Dataset / loader
    ds = CocoDetDataset(ANN_JSON, IMAGE_ROOT)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate,
                        pin_memory=torch.cuda.is_available())

    # Model
    model = load_model(len(ds.contig2catid), CKPT, device)

    # 1) Run detections once and save
    det_file = run_detections(model, ds, loader, device, OUT_DIR, SPLIT_NAME, score_thresh=SCORE_THRESH_SAVE)

    # 2) COCO eval overall
    coco_gt = ds.coco
    coco_dt = coco_gt.loadRes(det_file)
    coco_eval, _ = coco_eval_and_capture(coco_gt, coco_dt, OUT_DIR, SPLIT_NAME)

    # 3) Per-class & PR curves
    perclass_data = save_per_class_metrics(coco_eval, ds, OUT_DIR, SPLIT_NAME)
    save_pr_curves(perclass_data, OUT_DIR, SPLIT_NAME)

    # 4) Per-lighting & class×lighting
    if os.path.exists(SPLIT_TABLE):
        eval_per_lighting(coco_gt, coco_dt, ds, SPLIT_TABLE, OUT_DIR, SPLIT_NAME)
    else:
        print(f"WARNING: split table not found at {SPLIT_TABLE}; skipping lighting breakdowns.")

    # 5) Visual examples with TP/FP/FN coloring
    viz_dir = os.path.join(OUT_DIR, "viz")
    visualize_tp_fp_fn(model, ds, device, viz_dir, SPLIT_NAME,
                       num_images=VIZ_NUM_IMAGES, score_thr=VIZ_SCORE_THR,
                       iou_thr=VIZ_IOU_THR, seed=RANDOM_SEED)

if __name__ == "__main__":
    main()
