import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from dataset.bfov_framework import OmniFrameworkTorch


def generate_weight_map(mask, w_max=2, w_min=0.5, decay_power=0.5):
    weight_map = np.zeros_like(mask, dtype=np.float32)
    bg_mask = 1 - mask
    w_fore = bg_mask.sum() / mask.sum()
    if w_fore > w_max:
        w_fore = w_max
    elif w_fore < w_min:
        w_fore = w_min
    w_bg = 1 / w_fore
    weight_map[mask == 1] = w_fore
    dist_transform = cv2.distanceTransform(
        bg_mask.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=3
    )
    max_dist = np.max(dist_transform)
    if max_dist > 0:
        dist_norm = dist_transform / max_dist
    else:
        dist_norm = dist_transform
    bg_weight = w_bg + (w_fore - w_bg) * np.power(1 - dist_norm, decay_power)
    bg_indices = mask == 0
    weight_map[bg_indices] = bg_weight[bg_indices]
    return weight_map


@torch.no_grad()
def extract_weight(fake_image, omni_framework, binary_mask):
    if binary_mask.sum() != 0:
        try:
            _, mask_after_search = omni_framework.obtain_search_region(fake_image, binary_mask)
            search_weight = generate_weight_map(mask_after_search.cpu().numpy().astype(np.float32))
            _, out_weight, _ = omni_framework.reproject_search_region_to_weight(mask_after_search, search_weight)
            out_weight = out_weight.cpu()
            out_weight[out_weight == 0] = search_weight.min().item()
            return (mask_after_search, search_weight, out_weight)
        except:
            out_weight = torch.ones((fake_image.shape[0], fake_image.shape[1]))
            return (out_weight,)
    else:
        out_weight = torch.ones((fake_image.shape[0], fake_image.shape[1]))
        return (out_weight,)


class OmniVOSDataset(Dataset):
    def __init__(self, data_root, df_path, save_dir, device):
        self.data_root = data_root
        df = pd.read_csv(df_path)
        self.save_dir = save_dir
        self.device = device
        self.color_set = np.array([[128, 0, 0], [0, 128, 0], [0, 0, 128]])
        self.fake_image = np.zeros((1920, 3840, 3))
        self.video_data_list = []
        for index, row in df.iterrows():
            video_name = f"{row['Video']:03d}"
            obj_id = row['Object ID']
            index_tuple_list = eval(row["Info"])
            self.video_data_list.append((video_name, obj_id, index_tuple_list[0][0]))

    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        video_name, obj_id, start_id = self.video_data_list[idx]
        mask_dir = os.path.join(self.data_root, video_name, "mask")
        save_dir = os.path.join(self.save_dir, video_name, str(obj_id))
        os.makedirs(save_dir, exist_ok=True)
        omni_framework = OmniFrameworkTorch(ori_width=3840, ori_height=1920,
                                            out_width=3840, out_height=1920, device=self.device)
        mask_name_list = os.listdir(mask_dir)
        mask_name_list = sorted(mask_name_list, key=lambda x: int(x.split(".")[0]))[start_id:]
        for mask_name in mask_name_list:
            mask_path = os.path.join(mask_dir, mask_name)
            mask = Image.open(mask_path).convert("RGB")
            mask = np.array(mask)
            obj_color = self.color_set[obj_id]
            binary_mask = np.all(mask == obj_color, axis=-1)
            out_weight = extract_weight(self.fake_image, omni_framework, binary_mask)[-1]
            weight_save_path = os.path.join(save_dir, mask_name[:mask_name.index(".")] + ".npy")
            np.save(weight_save_path, out_weight)
        return 0


class PanoVOSDataset(Dataset):
    def __init__(self, data_root, df_path, save_dir, device):
        self.data_root = data_root
        df = pd.read_csv(df_path)
        self.save_dir = save_dir
        self.device = device
        self.color_set = np.array([[128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0], [0, 128, 128], [128, 0, 128], [128, 128, 128]])
        self.video_data_list = []
        for index, row in df.iterrows():
            video_name = f"{row['Video']}"
            obj_id = row['Object ID']
            index_tuple_list = eval(row["Info"])
            self.video_data_list.append((video_name, obj_id, index_tuple_list[0][0]))

    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        video_name, obj_id, start_id = self.video_data_list[idx]
        mask_dir = os.path.join(self.data_root, "Annotations", video_name)
        save_dir = os.path.join(self.save_dir, video_name, str(obj_id))
        os.makedirs(save_dir, exist_ok=True)
        mask_name_list = os.listdir(mask_dir)
        mask_name_list = sorted(mask_name_list, key=lambda x: int(x.split(".")[0]))[start_id:]
        omni_framework = None
        for mask_name in mask_name_list:
            mask_path = os.path.join(mask_dir, mask_name)
            mask = Image.open(mask_path).convert("RGB")
            mask = np.array(mask)
            fake_image = np.zeros((mask.shape[0], mask.shape[1], 3))
            if omni_framework is None:
                omni_framework = OmniFrameworkTorch(ori_width=mask.shape[1], ori_height=mask.shape[0],
                                                    out_width=mask.shape[1], out_height=mask.shape[0],
                                                    device=self.device)
            obj_color = self.color_set[obj_id]
            binary_mask = np.all(mask == obj_color, axis=-1)
            out_weight = extract_weight(fake_image, omni_framework, binary_mask)[-1]
            weight_save_path = os.path.join(save_dir, mask_name[:mask_name.index(".")] + ".npy")
            np.save(weight_save_path, out_weight)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mask weight maps for 360VOTS or PanoVOS dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--df_path", type=str, required=True, help="Path to CSV file (train.csv")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated weight maps")
    parser.add_argument("--dataset", type=str, required=True, choices=["360VOTS", "PanoVOS"], help="Dataset type: 360VOTS or PanoVOS")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == "360VOTS":
        dataset = OmniVOSDataset(args.data_root, args.df_path, args.save_dir, device)
    else:
        dataset = PanoVOSDataset(args.data_root, args.df_path, args.save_dir, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    pbar = tqdm(dataloader, total=len(dataset))
    for data in pbar:
        pass
