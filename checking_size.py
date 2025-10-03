import os
import csv
from PIL import Image

in_root  = "../Exdark/zero-dce++"   
out_root = "../Exdark/nafnet_all"   
manifest = "../Exdark/manifest.csv"

def is_img(fn): return fn.lower().endswith((".jpg",".jpeg",".png"))

rows = []
for root, _, files in os.walk(in_root):
    for fn in files:
        if not is_img(fn): 
            continue
        ip = os.path.join(root, fn)
        rel = os.path.relpath(ip, in_root)
        op = os.path.join(out_root, rel)

        try:
            iw, ih = Image.open(ip).size
        except Exception:
            iw, ih = -1, -1

        if os.path.isfile(op):
            try:
                ow, oh = Image.open(op).size
                status = "ok"
            except Exception:
                ow, oh = -1, -1
                status = "output_broken"
        else:
            ow, oh = -1, -1
            status = "missing_output"

        rows.append({
            "rel_path": rel,
            "old_w": iw, "old_h": ih,
            "new_w": ow, "new_h": oh,
            "status": status
        })

os.makedirs(out_root, exist_ok=True)
with open(manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["rel_path","old_w","old_h","new_w","new_h","status"])
    w.writeheader()
    w.writerows(rows)

print("Wrote manifest:", manifest)
print("Summary:",
      "total", len(rows),
      "resized", sum(1 for r in rows if r["status"]=="ok" and (r["old_w"]!=r["new_w"] or r["old_h"]!=r["new_h"])),
      "same_size", sum(1 for r in rows if r["status"]=="ok" and (r["old_w"]==r["new_w"] and r["old_h"]==r["new_h"])),
      "missing", sum(1 for r in rows if r["status"]!="ok"))
