import os
import json

image_root = '../../Exdark/zero-dce++'
json_paths = [
    '../../Exdark/Coco_output/exdark_train.json',
    '../../Exdark/Coco_output/exdark_val.json',
    '../../Exdark/Coco_output/exdark_test.json'
]

missing_files = []

for jp in json_paths:
    with open(jp, 'r') as f:
        data = json.load(f)
    for img in data['images']:
        img_path = os.path.join(image_root, img['file_name'])
        if not os.path.exists(img_path):
            missing_files.append(img_path)

if missing_files:
    print(f"Missing {len(missing_files)} images:")
    for m in missing_files[:20]:  # show first 20
        print("  ", m)
else:
    print("All JSON image paths exist in Zero-DCE++ folder.")
