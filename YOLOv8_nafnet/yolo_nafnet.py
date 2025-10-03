import os
import shutil

data_path = "../../Exdark/nafnet_all"               
imageclass = "../../imageclasslist.txt"            
label_path = "../../Exdark/yolov8_format"            
output_base = "../../Exdark/DATA_nafnet"           

split_map = {"1": "train", "2": "val", "3": "test"}

# Create folders for each split
for split in split_map.values():
    os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, "labels"), exist_ok=True)
with open(imageclass, 'r') as f:
    lines = f.readlines()
if lines[0].lower().startswith("name"):
    lines = lines[1:]

# Process each image
for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
        continue

    filename, _, _, _, split_type = parts
    split_name = split_map.get(split_type)

    if not split_name:
        print(f"Unknown split type '{split_type}' for {filename}")
        continue

    found = False
    for category in os.listdir(data_path): 
        category_path = os.path.join(data_path, category)
        if not os.path.isdir(category_path):
            continue

        src_img_path = os.path.join(category_path, filename)
        if os.path.exists(src_img_path):
            dst_img_path = os.path.join(output_base, split_name, "images", filename)
            shutil.copy2(src_img_path, dst_img_path)
            label_file = os.path.splitext(filename.lower())[0] + '.txt'
            src_label_path = os.path.join(label_path, label_file)
            dst_label_path = os.path.join(output_base, split_name, "labels", label_file)

            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
            else:
                print(f"Label missing for: {filename}")

            found = True
            break

    if not found:
        print(f"Image not found: {filename}")

print("Dataset restructuring complete!")
