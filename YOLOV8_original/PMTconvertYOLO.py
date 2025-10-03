import cv2
import os
from glob import glob

pmt_file = "../Exdark/ExDark_Annno"
image_file = "../Exdark/ExDark"
yolov_file = "../Exdark/yolov8_format"
labels = {
    'Bicycle': 0, 'Boat': 1, 'Bottle': 2, 'Bus': 3, 'Car': 4,
    'Cat': 5, 'Chair': 6, 'Cup': 7, 'Dog': 8, 'Motorbike': 9,
    'People': 10, 'Table': 11
}

os.makedirs(yolov_file, exist_ok=True)
missing_images = []
converted = 0
for class_folder in os.listdir(pmt_file):
    ann_path = os.path.join(pmt_file, class_folder)
    img_path = os.path.join(image_file, class_folder)
    if not os.path.isdir(ann_path):
        continue
    for ann_file in glob(os.path.join(ann_path, '*.txt')):
        base_name = os.path.splitext(os.path.basename(ann_file))[0]
        
        image_found = False
        # Look for image regardless of extension or case
        for file in os.listdir(img_path):
            if file.lower().startswith(base_name.lower()):
                image_path = os.path.join(img_path, file)
                base_name = os.path.splitext(file)[0]
                image_found = True
                break

                
        if not image_found:
            missing_images.append(f"{class_folder}/{base_name}")
            continue
            
        img= cv2.imread(image_path)
        if img is None:
            missing_images.append(f"{class_folder}/{base_name} (unreadable)")
            continue
            
        h_img, w_img = img.shape[:2]
        yolo_lines = []
        with open(ann_file, 'r') as f:
            for line in f:
                if line.startswith('%') or line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts)<5:
                    continue
                class_name = parts[0]
                
                class_id = labels[class_name]
                left,top,width,height = map(float, parts[1:5])
                x_center = (left+width/2)/w_img
                y_center = (top+height/2)/h_img
                w_norm = width/w_img
                h_norm = height/h_img
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        ### Save yolo labels
        out_file_path = os.path.join(yolov_file, base_name+'.txt')
        with open(out_file_path, 'w') as out_f:
            out_f.write('\n'.join(yolo_lines))
        converted +=1
        
print("\n_____Conversion Summary______")
print(f"Labels saved       : {converted}")
print(f"Missing images     : {len(missing_images)}")

if missing_images:
    print("\nExamples of missing images:")
    for item in missing_images[:10]:
        print(f"- {item}")

# Save label map
with open(os.path.join(yolov_file, "labels_map.txt"), "w") as f:
    for name, idx in labels.items():
        f.write(f"{idx}: {name}\n")        
                
                
    
    