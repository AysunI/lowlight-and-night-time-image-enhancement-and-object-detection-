import json
import os
import argparse
from collections import defaultdict

idtoclass = {
    1: "Bicycle", 2: "Boat", 3: "Bottle", 4: "Bus",
    5: "Car", 6: "Cat", 7: "Chair", 8: "Cup",
    9: "Dog", 10: "Motorbike", 11: "People", 12: "Table"}

def read_split_table(path, sep=None, has_header=True):
    split_by_name = {}
    class_by_name = {}
    with open(path, "r", errors="ignore") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if has_header and first:
                first = False
                if line.lower().startswith("name"):
                    continue
            parts = line.split(sep) if sep is not None else line.split()
            if len(parts) < 5:
                continue
            try:
                base = os.path.basename(parts[0])
                clss = int(parts[1])
                split_code = int(parts[4])
            except ValueError:
                continue
            split = {1: "train", 2: "val", 3: 'test'}.get(split_code)
            if split is None:
                continue
            split_by_name[base] = split
            class_by_name[base] = clss
    return split_by_name, class_by_name

def main():
    all_json = "../Exdark/exdark_coco_2.json"
    split_table = "../imageclasslist.txt"
    images_root = '../Exdark/ExDark'
    out_path = "../Exdark/Coco_output"
    sep = None
    no_header = False
    
    coco_all = json.load(open(all_json, 'r'))
    split_name, class_name = read_split_table(split_table, sep=sep, has_header=not no_header)
    print("rows parsed:", len(split_name))

    idtoimg = {im["id"]: im for im in coco_all['images']}
    basetoid = {os.path.basename(im['file_name']): im['id'] for im in coco_all["images"]}
    
    by_img = defaultdict(list)
    for a in coco_all.get("annotations", []):
        by_img[a["image_id"]].append(a)
    
    def fixed_image(im):
        base = os.path.basename(im['file_name'])
        cls_id = class_name.get(base)
        if cls_id in idtoclass:
            im = dict(im)
            im['file_name'] = f"{idtoclass[cls_id]}/{base}"   # forces folder by class
        return im  

    buckets = {'train': [], 'val': [], 'test': []}
    missing = []
    
    for base, split in split_name.items():
        img_id = basetoid.get(base)
        if img_id is None:
            missing.append(base)
            continue
        buckets[split].append((fixed_image(idtoimg[img_id]), by_img.get(img_id, [])))
    
    os.makedirs(out_path, exist_ok=True)
    
    for s, outname in [("train", "exdark_train.json"), ("val", "exdark_val.json"), ("test", "exdark_test.json")]:
        images = [im for im, _ in buckets[s]]
        annos = [a for _, anns in buckets[s] for a in anns]
        out = {'images': images, "annotations": annos, "categories": coco_all["categories"]}
        json.dump(out, open(os.path.join(out_path, outname), "w"))
        print(f"{s}: {len(images)} images, {len(annos)} annos -> {outname}")
        
    if missing:
        print("WARNING: in split table but not in COCO JSON:", missing[:10])

if __name__ == "__main__":
    main()
