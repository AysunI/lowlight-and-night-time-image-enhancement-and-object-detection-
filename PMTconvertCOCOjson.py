import os
import json
from PIL import Image

CLASSES = ["Bicycle","Boat","Bottle","Bus","Car","Cat","Chair","Cup","Dog","Motorbike","People","Table"]
NAME2ID = {n: i+1 for i, n in enumerate(CLASSES)}

ALIASES = {
    "bicycle":"Bicycle",
    "boat":"Boat",
    "bottle":"Bottle",
    "bus":"Bus",
    "car":"Car",
    "cat":"Cat",
    "chair":"Chair",
    "cup":"Cup","mug":"Cup",
    "dog":"Dog",
    "motorbike":"Motorbike",
    "people":"People",
    "table":"Table",
}

def _norm_label(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return ALIASES.get(s.lower(), s.title())

def _scan_images(image_root):
    """Build basename(lower) -> relative path under image_root."""
    mapping = {}
    for r, _, fs in os.walk(image_root):
        for f in fs:
            fl = f.lower()
            if fl.endswith((".jpg",".jpeg",".png")):
                rel = os.path.relpath(os.path.join(r, f), image_root).replace("\\","/")
                mapping[fl] = rel
    return mapping

def parse_pmt_line(line):
    """
    Returns (label or None, x, y, w, h) in COCO xywh.
    Accepts either:
      label x y w h   (xywh)
      x y w h         (xywh)
      label xmin ymin xmax ymax   (xyxy)
      xmin ymin xmax ymax         (xyxy)
    """
    line = line.strip()
    if not line or line.startswith('%'):
        return None
    parts = line.replace(",", " ").split()
    if len(parts) < 4:
        return None

    # label present?
    has_label = any(ch.isalpha() for ch in parts[0])
    if has_label:
        label = _norm_label(parts[0])
        nums = parts[1:5]
    else:
        label = None
        nums = parts[0:4]

    try:
        a, b, c, d = map(float, nums)
    except ValueError:
        return None

    # try to detect xyxy vs xywh
    if c > a and d > b:
        # could be xyxy
        x, y, w, h = a, b, c - a, d - b
    else:
        # assume xywh
        x, y, w, h = a, b, c, d

    if w <= 0 or h <= 0:
        return None
    return label, x, y, w, h

def convert_exdark_to_coco(image_root, anno_root, output_json):
    images = []
    annotations = []
    next_image_id = 1
    next_ann_id = 1

    # build a robust lookup of actual image locations
    base_to_rel = _scan_images(image_root)

    # fixed categories (official IDs)
    categories = [{"id": NAME2ID[name], "name": name} for name in CLASSES]

    missing_imgs = []
    missing_boxes = 0

    # Walk through annotation folders
    for root, _, files in os.walk(anno_root):
        rel_class_dir = os.path.relpath(root, anno_root)  # e.g., 'Bottle'
        if rel_class_dir == ".":
            continue
        default_label = _norm_label(os.path.basename(rel_class_dir))

        for file in files:
            if not file.lower().endswith(".txt"):
                continue

            # derive image basename (without extension)
            stem = file[:-4]
            # some datasets name txt as "xxx.jpg.txt" -> strip twice
            if stem.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                stem = os.path.splitext(stem)[0]

            # find the real image path under image_root (any ext, any folder)
            rel = None
            for ext in (".jpg",".jpeg",".png",".bmp"):
                rel = base_to_rel.get((stem + ext).lower())
                if rel:
                    break
            if rel is None:
                # last try: assume it sits next to the class dir
                candidate = os.path.join(image_root, rel_class_dir, stem + ".jpg")
                if os.path.exists(candidate):
                    rel = os.path.relpath(candidate, image_root).replace("\\","/")
            if rel is None:
                missing_imgs.append(os.path.join(rel_class_dir, stem))
                continue

            abs_path = os.path.join(image_root, rel)
            with Image.open(abs_path) as img:
                width, height = img.size

            image_id = next_image_id
            next_image_id += 1
            images.append({
                "id": image_id,
                "file_name": rel,   
                "width": width,
                "height": height
            })

            # Read annotation file
            with open(os.path.join(root, file), "r", errors="ignore") as f:
                for line in f:
                    parsed = parse_pmt_line(line)
                    if parsed is None:
                        continue
                    label, x, y, w, h = parsed
                    if label is None:
                        label = default_label
                    label = _norm_label(label)
                    cat_id = NAME2ID.get(label)
                    if cat_id is None:
                        continue

                    annotations.append({
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": int(cat_id),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w*h),
                        "iscrowd": 0
                    })
                    next_ann_id += 1
                else:
                    pass

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"COCO JSON saved to {output_json}")
    print(f"Total images: {len(images)}, annotations: {len(annotations)}, categories: {len(categories)}")
    print("Category ID map:", {c["id"]: c["name"] for c in categories})
    if missing_imgs:
        print(f"WARNING: {len(missing_imgs)} images listed by annotations not found under image_root (by name/ext). "
              f"Example: {missing_imgs[:5]}")

image_root = "../Exdark/ExDark"        
anno_root  = "../Exdark/ExDark_Annno"   
output_json = "../Exdark/exdark_coco_2.json"

convert_exdark_to_coco(image_root, anno_root, output_json)
