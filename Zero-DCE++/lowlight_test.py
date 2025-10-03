import torch
import torch.nn as nn
import torchvision
import os
import time
import model
import numpy as np
from PIL import Image
import glob


def lowlight(image_path, output_path):
    scale_factor = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess image
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]

    data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)

    # Load model
    DCE_net = model.enhance_net_nopool(scale_factor).to(device)
    DCE_net.load_state_dict(torch.load("snapshots_Zero_DCE++/Epoch99.pth", map_location=device))
    DCE_net.eval()

    # Inference
    start = time.time()
    enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start
    print("Inference time: {:.2f} s".format(end_time))


    # Save enhanced image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, output_path)

    return end_time

if __name__ == "__main__":
    input_images = "../../Exdark/ExDark"
    output_images = "../../Exdark/zero-dce++"
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    total_time = 0

    for category in os.listdir(input_images):
        cat_path = os.path.join(input_images, category)
        if not os.path.isdir(cat_path):
            continue

        print("Processing folder:", category)
        for ext in extensions:
            image_list = glob.glob(os.path.join(cat_path, ext))
            for image_path in image_list:
                rel_path = os.path.relpath(image_path, input_images)
                output_path = os.path.join(output_images, rel_path)
                if os.path.exists(output_path):
                    print(f"Skipping {output_path}, already processed.")
                    continue
                total_time += lowlight(image_path, output_path)

    print(" Total processing time: {:.2f} seconds.".format(total_time))
