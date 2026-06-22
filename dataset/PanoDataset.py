import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import random
import torch.nn.functional as F

class PanoVOSDataset(Dataset):
    def __init__(self, df_root, data_root, weight_root, video_size=20, sample_num=None, val=False):
        self.data_df = pd.read_csv(df_root)
        self.data_root = data_root
        self.weight_root = weight_root
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.video_size = video_size
        self.data_index_tuple_list = []
        self.color_set = np.array([[128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0], [0, 128, 128], [128, 0, 128], [128, 128, 128]])
        self.sample_num = sample_num
        self.val = val
        self.initiate_data()

    def initiate_data(self):
        pbar = tqdm(self.data_df.iterrows(), desc="Load Data", leave=True, total=len(self.data_df))
        for index, row in pbar:
            video_name = row['Video']
            obj_id = row['Object ID']
            index_tuple_list = eval(row["Info"])
            frame_num = len(os.listdir(os.path.join(self.data_root, "JPEGImages", video_name)))
            for index_tuple in index_tuple_list:
                if index_tuple[0] == index_tuple[1]:
                    continue
                if frame_num - 1 < index_tuple[0] + self.video_size:
                    continue
                self.data_index_tuple_list.append((video_name, obj_id, index_tuple[0], index_tuple[1], frame_num))

    def __len__(self):
        return self.sample_num if self.sample_num is not None else len(self.data_index_tuple_list)

    @staticmethod
    def load_image(img_path):
        return Image.open(img_path).convert("RGB")

    def load_weight(self, weight_path):
        if os.path.exists(weight_path) and os.path.exists(weight_path):
            weight = np.load(weight_path)
            weight = torch.from_numpy(weight.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            weight[weight > 2] = 2
            weight[weight < 0.5] = 0.5
            weight = F.interpolate(weight, (1024, 1024), mode="bilinear", align_corners=False)
        else:
            weight = torch.ones((1, 1, 1024, 1024)).float()
        return weight

    @torch.no_grad()
    def load_mask(self, mask_path, obj_id):
        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask)
        obj_color = self.color_set[obj_id]
        binary_mask = np.all(mask == obj_color, axis=-1)
        binary_mask = torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        binary_mask = F.interpolate(binary_mask, (1024, 1024), mode="bilinear", align_corners=False) > 0.5
        return binary_mask


    def __getitem__(self, idx):
        if self.sample_num is not None:
            idx = random.randint(0, len(self.data_index_tuple_list) - 1)
        video_name, obj_id, ori_start_index, end_index, frame_num = self.data_index_tuple_list[idx]
        if not self.val:
            start_index = ori_start_index
            while end_index != start_index + self.video_size:
                start_index = random.randint(ori_start_index, end_index - 1)
                end_index = start_index + self.video_size
                if end_index > frame_num - 1:
                    end_index = frame_num - 1
        else:
            start_index = random.randint(ori_start_index, end_index - 1)
            end_index = start_index + self.video_size
            if end_index > frame_num - 1:
                end_index = frame_num - 1
        video_path = os.path.join(self.data_root, "JPEGImages", video_name)
        annotation_path = os.path.join(self.data_root, "Annotations", video_name)
        weight_dir_path = os.path.join(self.weight_root, video_name, str(obj_id))
        file_start_index = min([int(file_name[:file_name.index(".")]) for file_name in os.listdir(video_path)])
        frames = []
        masks = []
        weights = []

        for i in range(start_index, end_index + 1):
            file_index = file_start_index + 5 * i
            img_path = os.path.join(video_path, f"{file_index:05d}.jpg")
            img = self.load_image(img_path)
            ori_h, ori_w = img.size[:2]
            frame = self.image_transform(img)
            frames.append(frame)
            mask_path = os.path.join(annotation_path, f"{file_index:05d}.png")
            weight_path = os.path.join(weight_dir_path, f"{file_index:05d}.npy")
            mask = self.load_mask(mask_path, obj_id)
            weight = self.load_weight(weight_path)
            masks.append(mask)
            weights.append(weight)

        frames = torch.stack(frames)
        masks = torch.cat(masks).float()
        weights = torch.cat(weights)
        if self.val:
            return {"frames": frames, "masks": masks, "shape": (ori_w, ori_h)}
        return {"frames": frames, "masks": masks, "weights": weights}

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys
    dataset = PanoVOSDataset(
        df_root="../../../Dataset/PanoVOS/train.csv",
        data_root="../../../Dataset/PanoVOS/train",
        weight_root="../../../Dataset/PanoVOS/train_mask_weight"
    )
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=1)
    for data in dataloader:
        print(data["frames"].shape) # [1, 101, 3, 1024, 1024]
        print(data["masks"].shape) # [1, 101, 1, 1024, 1024]
        print(data["weights"].shape) # [1, 101, 1, 1024, 1024]
        sys.exit(0)
