import shutil
import torch
import cv2
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
from config import Config
from model import PanoSAM2, init_crf, crf_refine
import torch.nn.functional as F

img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    ori_h, ori_w = image.size[:2]
    image_ = np.array(image)
    image_np = np.array(image.resize((1024, 1024))) / 255
    torch_image = torch.from_numpy(image_np).permute(2, 0, 1).to(torch.float)
    torch_image -= img_mean
    torch_image /= img_std

    return image_, ori_h, ori_w, torch_image.unsqueeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PanoSAM2 inference on a custom video with an initial mask")
    parser.add_argument("--video_frame_dir", type=str, required=True, help="Path to the directory of video frames")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to the initial mask (default: <video_frame_dir>.png)")
    parser.add_argument("--ckpt_path", type=str, default=f"result/weight/360VOTS/{Config.backbone}/PanoSAM2_40_{Config.backbone}.pth", help="Path to model checkpoint")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save predicted masks (default: result/<video_name>_<backbone>)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--no_long_mem", action="store_true", help="Disable long-term memory (LSM)")
    args = parser.parse_args()

    mask_path = args.mask_path if args.mask_path is not None else args.video_frame_dir + ".png"
    save_dir = args.save_dir if args.save_dir is not None else os.path.join(
        "result/visualization/Self-Collected/PanoSAM2",
        f"{args.video_frame_dir[args.video_frame_dir.rindex('/')+1:]}_{Config.backbone}")
    use_long_mem = not args.no_long_mem

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = PanoSAM2(Config).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    crf, crf_band_px, crf_p_core = init_crf(device)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    def save_mask(image, mask, ori_h, ori_w, frame_path):
        image = torch.from_numpy(image).permute(2, 0, 1).to(device)
        mask = F.interpolate(mask, size=(ori_w, ori_h), mode="bilinear") > 0
        mask = crf_refine(crf, crf_band_px, crf_p_core, image, mask[0, 0])
        mask = mask.cpu().numpy().astype(np.uint8) * 255
        save_path = os.path.join(save_dir, os.path.basename(frame_path))
        cv2.imwrite(save_path, mask)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        frame_name_list = os.listdir(args.video_frame_dir)
        frame_name_list = sorted(frame_name_list, key=lambda x: int(x[:x.rindex(".")]))
        frame_path_list = [os.path.join(args.video_frame_dir, frame_name)
                           for frame_name in frame_name_list]

        ori_image, ori_h, ori_w, first_frame = preprocess_image(frame_path_list[0])
        first_frame = first_frame.to(device)
        model.process_first_frame(first_frame, mask)
        save_mask(ori_image, mask, ori_h, ori_w, frame_path_list[0])
        frame_index = 1
        for frame_path in tqdm(frame_path_list[1:]):
            ori_image, _, _, frame = preprocess_image(frame_path)
            frame = frame.to(device)
            masks_logit = model.process_frame(frame.to(device), frame_index, use_long_mem)
            save_mask(ori_image, masks_logit, ori_h, ori_w, frame_path)
            frame_index += 1
