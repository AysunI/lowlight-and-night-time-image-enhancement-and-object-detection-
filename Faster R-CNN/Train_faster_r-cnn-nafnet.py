import os 
import json
import time
import math
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm 

image_root = '../../Exdark/nafnet_all'
train_json = '../../Exdark/resized_coco_json/instances_train.json'
val_json = '../../Exdark/resized_coco_json/instances_val.json'
test_json = '../../Exdark/resized_coco_json/instances_test.json'
out_path = "../../Exdark/exdark_frcnn_outputs_nafnet"
epochs = 12
USE_AMP = True

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class CocoDetDataset(Dataset):
    def __init__(self, ann_file, img_root, train=True):
        self.coco = COCO(ann_file)
        self.img_root = img_root
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.train = train
        
        # Build id mappings
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats_sorted = sorted(cats, key=lambda c: c['id'])
        self.catid2contig = {c['id']: i+1 for i, c in enumerate(cats_sorted)}
        self.contig2catid = {i+1: c['id'] for i, c in enumerate(cats_sorted)}
        self.contig2name  = {i+1: c['name'] for i, c in enumerate(cats_sorted)}
        self.num_classes  = len(cats_sorted)
        
        if train:
            self.tf = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor()
            ])
        else:
            self.tf = transforms.ToTensor()
            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_root, info['file_name'])
        img = Image.open(path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, iscrowd, area = [], [], [], []
        for a in anns:
            if "bbox" not in a:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid2contig[a['category_id']])
            iscrowd.append(a.get("iscrowd", 0))
            area.append(a.get('area', w * h))

        img = self.tf(img)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32) if area else torch.zeros((0,), dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return img, target

def collate(batch):
    return tuple(zip(*batch))

def build_model(num_classes):
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    except Exception:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")

    # Smaller anchors can help on smaller/low-light objects
    from torchvision.models.detection.rpn import AnchorGenerator
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feats, 1 + num_classes
    )
    return model

@torch.no_grad()
def evaluate_coco(model, dataset, loader, device, out_dir, split_name="val"):
    """Runs COCO eval and returns AP@[.50:.95] as a float."""
    model.eval()
    results = []
    for imgs, targets in tqdm(loader, desc=f"Evaluating ({split_name})"):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for out, t in zip(outputs, targets):
            img_id = int(t["image_id"].item())
            boxes = out["boxes"].detach().cpu().tolist()
            scores = out["scores"].detach().cpu().tolist()
            labels = out["labels"].detach().cpu().tolist()
            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                cat_id = int(dataset.contig2catid[int(l)])
                results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s)
                })

    if not results:
        print("No detections produced.")
        return 0.0

    os.makedirs(out_dir, exist_ok=True)
    res_file = os.path.join(out_dir, f"coco_{split_name}_detections.json")
    json.dump(results, open(res_file, "w"))

    coco_dt = dataset.coco.loadRes(res_file)
    coco_eval = COCOeval(dataset.coco, coco_dt, "bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    ap = float(coco_eval.stats[0]) if coco_eval.stats is not None else 0.0  # AP@[.50:.95]
    return ap

def save_ckpt(path, model, num_classes, contig2name):
    torch.save({
        "model": model.state_dict(),
        "num_classes": num_classes,
        "contig2name": contig2name
    }, path)

def main():
    os.makedirs(out_path, exist_ok=True)

    # Datasets 
    train_ds = CocoDetDataset(train_json, image_root, train=True)
    val_ds   = CocoDetDataset(val_json,   image_root, train=False)
    print("Classes:", train_ds.num_classes, train_ds.contig2name)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False,
                              num_workers=4, collate_fn=collate, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(train_ds.num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    best_ap = -1.0
    best_path = os.path.join(out_path, "model_best.pth")

    # ---- Train ----
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        running = 0.0
        for imgs, targets in pbar:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    loss_dict = model(imgs, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            running = 0.9 * running + 0.1 * float(loss.item()) if running else float(loss.item())
            pbar.set_postfix(loss=f"{running:.3f}")

        lr_scheduler.step()

        # ---- Validate this epoch ----
        val_ap = evaluate_coco(model, val_ds, val_loader, device, out_dir=out_path, split_name=f"val_ep{epoch:02d}")
        print(f"[Epoch {epoch}] Val AP@[.50:.95]: {val_ap:.4f} | Best so far: {best_ap:.4f}")

        # ---- Save best so far ----
        if val_ap > best_ap:
            best_ap = val_ap
            save_ckpt(best_path, model, train_ds.num_classes, train_ds.contig2name)
            print(f"✔ Saved new BEST to {best_path}")

    # ---- Save last ----
    last_path = os.path.join(out_path, "model_last.pth")
    save_ckpt(last_path, model, train_ds.num_classes, train_ds.contig2name)
    print(f"Saved LAST to {last_path}")

    # ---- final test eval with best model ----
    if test_json and os.path.exists(test_json):
        # load best for test (safer)
        print("\nEvaluating BEST on TEST set…")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_ds = CocoDetDataset(test_json, image_root, train=False)
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False,
                                 num_workers=4, collate_fn=collate, pin_memory=True)
        test_ap = evaluate_coco(model, test_ds, test_loader, device, out_dir=out_path, split_name="test_best")
        print(f"TEST AP@[.50:.95]: {test_ap:.4f}")

if __name__ == "__main__":
    main()
