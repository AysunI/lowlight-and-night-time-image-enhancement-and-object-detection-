import os, math, yaml
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from NAFNet_arch import NAFNet


yaml_path      = "NAFNet-width64.yml"
weights_path   = "NAFNet-SIDD-width64.pth"
in_root        = "../../Exdark/zero-dce++"     
out_root       = "../../Exdark/nafnet_all"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# -------- Load Config & Model --------
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

model = NAFNet(
    img_channel=3,
    width=cfg['network_g']['width'],
    middle_blk_num=cfg['network_g']['middle_blk_num'],
    enc_blk_nums=cfg['network_g']['enc_blk_nums'],
    dec_blk_nums=cfg['network_g']['dec_blk_nums']
).to(device)

ckpt = torch.load(weights_path, map_location=device)
state = ckpt.get("params", ckpt.get("state_dict", ckpt))
missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()

# -------- Helpers --------
def pil_to_tensor(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    arr = arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0
    return arr

def tensor_to_pil(t):
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")

def pad_to_multiple(x, multiple=64):
    _, _, h, w = x.shape
    H = math.ceil(h / multiple) * multiple
    W = math.ceil(w / multiple) * multiple
    ph, pw = H - h, W - w
    if ph == 0 and pw == 0:
        return x, (h, w)
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)

def is_image_file(fn):
    return fn.lower().endswith((".jpg", ".jpeg", ".png"))

# -------- Sanity Check --------
print("\n--- Running sanity check ---")
x_test = torch.ones(1, 3, 256, 256, device=device) * 0.5
with torch.inference_mode():
    y_test = model(x_test.float())
print(f"Sanity check output range: min={y_test.min().item():.4f}, max={y_test.max().item():.4f}")

# -------- Process All Images --------
def process_image(ipath, opath):
    img = ImageOps.exif_transpose(Image.open(ipath).convert("RGB"))
    ten = pil_to_tensor(img).unsqueeze(0).to(device)
    ten, (h, w) = pad_to_multiple(ten, 64)

    with torch.inference_mode():
        out = model(ten.float())[:, :, :h, :w]
        out = torch.clamp(out, 0.0, 1.0)  # Prevent rainbow artifacts

    out_img = tensor_to_pil(out)
    os.makedirs(os.path.dirname(opath), exist_ok=True)
    out_img.save(opath)

def process_folder(in_dir, out_dir):
    for root, dirs, files in os.walk(in_dir):
        for fn in files:
            if not is_image_file(fn):
                continue
            ip = os.path.join(root, fn)
            rel_path = os.path.relpath(ip, in_dir)
            op = os.path.join(out_dir, rel_path)
            process_image(ip, op)
            print(f"Saved {op}")

if __name__ == "__main__":
    process_folder(in_root, out_root)
    print("\nAll images processed and saved to", out_root)
