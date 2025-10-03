import json
import csv
import os 

manifest_path = '../Exdark/manifest.csv'
json_files = {
    "train": "../Exdark/Coco_output/exdark_train.json",
    "test":  "../Exdark/Coco_output/exdark_test.json",
    "val":   "../Exdark/Coco_output/exdark_val.json"
}

out_dir = "../Exdark/resized_coco_json"


def load_manifest(path):
    d = {}
    with open(path,"r") as f:
        r = csv.DictReader(f)
        for row in r :
            fname = row['rel_path']
            d[fname] = {
                "old_w": int(row['old_w']),
                "old_h": int(row['old_h']),
                "new_w": int(row['new_w']),
                "new_h": int(row['new_h'])
            }
    return d

def resize_json(in_json, out_json, manifest):
    with open(in_json, "r") as f:
        coco = json.load(f)
    for im in coco['images']:
        fname = im['file_name']
        if fname not in manifest:
            continue
        info = manifest[fname]
        if info['old_w'] == info['new_w'] and info['old_h']==info['new_h']:
            continue
            
        scale_x = info['new_w']/info['old_w']
        scale_y = info['new_h']/info['old_h']
        
        #### update image size 
        im['width'] = info['new_w']
        im['height'] = info['new_h']
        
        #### update annotations
        for ann in coco['annotations']:
            if ann['image_id']!=im['id']:
                continue
            x,y,w,h = ann['bbox']
            ann['bbox']=[x*scale_x, y*scale_y, w*scale_x, h*scale_y]
            if 'segmentation' in ann and ann['segmentation']:
                new_segs = []
                for seg in ann['segmentation']:
                    new_segs.append([
                        seg[i]*scale_x if i%2==0 else seg[i]*scale_y
                        for i in range(len(seg))])
                ann['segmentation']=new_segs 
                
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco,f)
    print('saved', out_json)
    
if __name__=='__main__':
    manifest = load_manifest(manifest_path)
    
    for split, in_json in json_files.items():
        out_json = os.path.join(out_dir, f'instances_{split}.json')
        resize_json(in_json,out_json,manifest)
