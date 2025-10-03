import zipfile
import os

zip_path = "../Exdark/DATA_original_yolo.zip"
extract_dir = "../Exdark/DATA_original"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Files extracted to: {extract_dir}")
