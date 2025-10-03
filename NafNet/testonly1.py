# In this code I have checked just one image 
import math, yaml, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from NAFNet_arch import NAFNet

yaml_path     = "NAFNet-width64.yml"
weights_path  = "NAFNet-SIDD-width64.pth"
in_image_path = "../../Exdark/zero-dce++/Cat/2015_03138.jpg"   
out_image_path= "../../Exdark/nafnet_pipeline_out.jpg"

# --------- Options ----------
USE_OPENCV_DENOISE = True     # set False to skip pre-denoise
OPENCV_H_LUMA      = 5        # 3–10 (mild)
OPENCV_H_CHROMA    = 5        # 3–10 (mild)
OPENCV_TEMPLATE    = 7
OPENCV_SEARCH      = 21

ALPHA_BLEND        = 0.75     # 0.6–0.8 keeps texture
APPLY_UNSHARP      = False    # tiny crisp boost (optional)
UNSHARP_RADIUS     = 1.0
UNSHARP_PERCENT    = 60
UNSHARP_THRESH     = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# --------- Try OpenCV ----------
cv2 = None
if USE_OPENCV_DENOISE:
    try:
        import cv2
    except Exception as e:
        print("OpenCV not available, continuing without pre-denoise:", e)
        USE_OPENCV_DENOISE = False

# --------- Model from YAML ----------
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

model = NAFNet(
    img_channel=3,
    width=cfg['network_g']['width'],
    middle_blk_num=cfg['network_g']['middle_blk_num'],
    enc_blk_nums=cfg['network_g']['enc_blk_nums'],
    dec_blk_nums=cfg['network_g']['dec_blk_nums'],
).to(device).eval()

ckpt = torch.load(weights_path, map_location=device)
state = ckpt.get("params", ckpt.get("state_dict", ckpt))
model.load_state_dict(state, strict=False)

# --------- Helpers ----------
def pil_to_tensor(img):
    if img.mode != "RGB": img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)             # HxWx3
    return torch.from_numpy(arr).permute(2,0,1).contiguous().float() / 255.0

def tensor_to_pil_safe(t):
    if t.dim()==4: t = t.squeeze(0)
    t = t.clamp(0,1).to(torch.float32).cpu().permute(1,2,0).contiguous().numpy()
    return Image.fromarray((t*255.0).round().astype(np.uint8), mode="RGB")

def pad64(x):
    _,_,h,w = x.shape
    H = math.ceil(h/64)*64; W = math.ceil(w/64)*64
    return F.pad(x,(0,W-w,0,H-h),mode="reflect"), (h,w)

def sanitize_in(t):
    eps = 1.0/255.0                     # soft clip to avoid extremes
    t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
    return t.clamp(eps, 1.0-eps)

def sanitize_out(t):
    t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
    return t.clamp(0.0, 1.0)

@torch.inference_mode()
def nafnet_forward_fp32(x):
    if device.type == "cuda":
        with torch.cuda.amp.autocast(enabled=False):
            y = model(x.float())
    else:
        y = model(x.float())
    return y.float()

# --------- Run one image ----------
with torch.inference_mode():
    # 1) Load Zero-DCE++ result
    img = ImageOps.exif_transpose(Image.open(in_image_path).convert("RGB"))

    # 2) Mild denoise in sRGB to stabilize NAFNet input
    if USE_OPENCV_DENOISE and cv2 is not None:
        img_bgr = np.array(img)[:, :, ::-1]  # RGB->BGR
        den = cv2.fastNlMeansDenoisingColored(
            img_bgr,
            None,
            h=OPENCV_H_LUMA,
            hColor=OPENCV_H_CHROMA,
            templateWindowSize=OPENCV_TEMPLATE,
            searchWindowSize=OPENCV_SEARCH
        )
        img = Image.fromarray(den[:, :, ::-1])  # back to RGB

    ten_in = pil_to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]
    print(f"in:  min={float(ten_in.min()):.4f}, max={float(ten_in.max()):.4f}, finite={bool(torch.isfinite(ten_in).all())}")

    # 3) Sanitize + pad
    ten = sanitize_in(ten_in)
    x,(h,w) = pad64(ten)

    # 4) NAFNet (fp32)
    out = nafnet_forward_fp32(x)[:, :, :h, :w]
    out = sanitize_out(out)
    print(f"nafnet: min={float(out.min()):.4f}, max={float(out.max()):.4f}")

    # 5) Blend with original (keep detail)
    alpha = float(ALPHA_BLEND)
    alpha = max(0.0, min(1.0, alpha))
    blended = (alpha * out + (1.0 - alpha) * ten_in).clamp(0,1)
    print(f"blend : min={float(blended.min()):.4f}, max={float(blended.max()):.4f}")

# 6) Optional light sharpen and save
img_out = tensor_to_pil_safe(blended)
if APPLY_UNSHARP:
    img_out = img_out.filter(ImageFilter.UnsharpMask(
        radius=UNSHARP_RADIUS, percent=UNSHARP_PERCENT, threshold=UNSHARP_THRESH
    ))

img_out.save(out_image_path)
print("saved:", out_image_path)
