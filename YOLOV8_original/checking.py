import json
import os 

gt_json = '../../Exdark/Coco_output/exdark_test.json'
dt_json = 'runs/detect/val2/predictions.json'

gt = json.load(open(gt_json))
dt = json.load(open(dt_json))

gt_ids = {im["id"] for im in gt["images"]}
dt_ids = {d["image_id"] for d in dt}

print("GT image_id count:", len(gt_ids))
print("DT image_id count:", len(dt_ids))
print("Overlap ids:", len(gt_ids & dt_ids))
print("Only in DT (first 5):", list(dt_ids - gt_ids)[:5])
file2id = {os.path.basename(im["file_name"]): im["id"] for im in gt["images"]}
fixed = False
for d in dt:
    if isinstance(d["image_id"], str):
        fn = os.path.basename(d["image_id"])
        if fn in file2id:
            d["image_id"] = file2id[fn]
            fixed = True

if fixed:
    json.dump(dt, open("runs/detect/val2/predictions_fixed.json","w"))
    print("Wrote fixed:", "runs/detect/val2/predictions_fixed.json")