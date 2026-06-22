import sys
import torch
from config import Config, omni_dataset_dict, pano_dataset_dict
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
import os
import numpy as np
from dataset import OmniDataset, PanoDataset
from metric import train_val_iou_measure, calculate_mean_iou_boundary_accuracy
from model import PanoSAM2
import torch.optim as optim
from tqdm import tqdm
from utils.loss_mask import loss_masks, iou_loss_jit, occlusion_loss
import torch.nn.functional as F
import random

random.seed(42)

os.makedirs(Config.save_path, exist_ok=True)


def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def train():
    rank, world_size = setup_distributed()
    model = PanoSAM2(Config)
    if Config.pretrain_path is not None:
        model.load_state_dict(torch.load(Config.pretrain_path), strict=False)
        print(f"\nLoad pretrained weight: {Config.pretrain_path}")
    model = model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False,
                                                find_unused_parameters=True)
    if Config.dataset == "360VOTS":
        train_dataset = OmniDataset.OmniVOSTrainDataset(omni_dataset_dict["train"]["df_root"],
                                                        omni_dataset_dict["train"]["data_root"],
                                                        omni_dataset_dict["train"]["weight_root"],
                                                        video_size=Config.video_size, sample_num=Config.train_sample_num)
        val_dataset = OmniDataset.OmniVOSValDataset(omni_dataset_dict["val"]["df_root"],
                                                    omni_dataset_dict["val"]["data_root"],
                                                    video_size=Config.video_size, sample_num=Config.test_sample_num)

    elif Config.dataset == "PanoVOS":
        train_dataset = PanoDataset.PanoVOSDataset(pano_dataset_dict["df_root"],
                                                   pano_dataset_dict["data_root"],
                                                   pano_dataset_dict["weight_root"],
                                                   video_size=Config.video_size,
                                                   sample_num=Config.train_sample_num, val=False)
        val_dataset = PanoDataset.PanoVOSDataset(pano_dataset_dict["df_root"],
                                                 pano_dataset_dict["data_root"],
                                                 pano_dataset_dict["weight_root"],
                                                 video_size=Config.video_size,
                                                 sample_num=Config.test_sample_num, val=True)
    else:
        dist.destroy_process_group()
        raise "Unknown dataset"
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, sampler=train_sampler,
                                  num_workers=Config.num_workers)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=Config.num_workers)
    step = 0
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60000])
    for epoch in range(1, Config.epoch_num + 1):
        correct_batch = None
        epoch_sum = 0
        for correct_batch_, correct_epoch in Config.correct_epoch_info:
            epoch_sum += correct_epoch
            if correct_epoch == -1 or epoch <= epoch_sum:
                correct_batch = correct_batch_
                break

        use_long_mem = (epoch + 1) > Config.mem_start_epoch
        print(f"\nCorrect the mask per {correct_batch} images. Train long memory: {use_long_mem}.")

        train_sampler.set_epoch(epoch)
        model.train()

        pbar = tqdm(train_dataloader, total=len(train_dataset) // (world_size * Config.batch_size),
                    desc=f"Train Epoch {epoch} Rank {rank}")
        for train_data in pbar:
            train_data["frames"] = train_data["frames"].cuda(rank)
            train_data["masks"] = train_data["masks"].cuda(rank)
            train_data["weights"] = train_data["weights"].cuda(rank)
            images, masks, weights = train_data["frames"], train_data["masks"], train_data["weights"]
            frame_index_list = list(range(1, images.shape[1]))
            groups = [frame_index_list[i:i + Config.group_frame_num] for i in
                      range(0, len(frame_index_list), Config.group_frame_num)]
            for batch_index in range(images.shape[0]):
                model.module.process_first_frame(images[[batch_index], 0], masks[[batch_index], 0], batch_index)
            pred_masks = []
            input_masks = masks
            weights = (F.interpolate(weights.squeeze(2), size=(256, 256), mode="bilinear")).unsqueeze(2)
            masks = ((F.interpolate(masks.squeeze(2), size=(256, 256), mode="bilinear") > 0.5).float()).unsqueeze(2)
            loss_list = []
            for group in groups:
                total_loss = 0
                for frame_idx in group:
                    pred_masks_ = []
                    for batch_index in range(len(images)):
                        sam2_pred_masks_logits, pred_ious, occlusion_score_logit = model.module.process_frame(
                            images[[batch_index], frame_idx], frame_idx,
                            use_long_mem=use_long_mem,
                            gt_mask=input_masks[[batch_index], frame_idx], correct_freq=correct_batch,
                            batch_index=batch_index)
                        best_iou_ind = pred_ious.argmax(dim=-1)
                        sam2_mask_logit = sam2_pred_masks_logits[:, best_iou_ind]
                        iou_loss = iou_loss_jit(sam2_pred_masks_logits.detach().transpose(0, 1),
                                                masks[batch_index, [frame_idx, frame_idx, frame_idx]], pred_ious, 3)
                        pred_masks_.append(sam2_mask_logit > 0)
                        loss_mask_sam, loss_dice_sam = loss_masks(sam2_mask_logit, masks[[batch_index], frame_idx], 1,
                                                                  weight_map=weights[[batch_index], frame_idx])
                        loss_occlusion = occlusion_loss(masks[[batch_index], frame_idx], occlusion_score_logit)
                        loss = iou_loss +  (0.5 * loss_mask_sam + 0.5 * loss_dice_sam) + 0.1 * loss_occlusion
                        total_loss = loss + total_loss
                    pred_masks_ = torch.cat(pred_masks_, 0)
                    pred_masks.append(pred_masks_)
                optimizer.zero_grad()
                total_loss = total_loss / (images.shape[0] * len(group))
                total_loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.max_norm)
                optimizer.step()
                torch.cuda.empty_cache()
                loss_list.append(total_loss.item())

            if step % Config.log_freq == 0:
                pred_masks = torch.stack(pred_masks, 1)
                mask_iou = train_val_iou_measure(pred_masks, masks[:, 1:])
                print(
                    f"\nTrain Epoch {epoch}, Step {step}, Rank {rank}, Loss: {np.mean(loss_list)}, IoU: {mask_iou}")
                torch.cuda.empty_cache()
            step += 1
        lr_scheduler.step()

        val_sampler.set_epoch(epoch)
        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dataloader, total=len(val_dataset) // world_size,
                        desc=f"Test Epoch {epoch} Rank {rank}")
            iou = 0
            boundary = 0
            overall = 0
            for val_data in pbar:
                frames, masks = val_data["frames"].cuda(rank), val_data["masks"]
                video_shape = (val_data["shape"][0].item(), val_data["shape"][1].item())
                model.module.process_first_frame(frames[:, 0], masks[:, 0].cuda(rank))

                frame_index_list = list(range(1, frames.shape[1]))
                pred_masks = []
                for frame_idx in frame_index_list:
                    mask_logit = model.module.process_frame(frames[:, frame_idx], frame_idx, use_long_mem)
                    pred_masks.append(mask_logit.cpu())
                    torch.cuda.empty_cache()
                pred_masks = torch.cat(pred_masks, 0)
                pred_masks = F.interpolate(pred_masks[::Config.eval_sample].float(), size=video_shape,
                                           mode="bilinear") > 0
                pred_masks = pred_masks.squeeze(1)
                masks = F.interpolate(masks[0, 1::Config.eval_sample], size=video_shape, mode="bilinear") > 0.5
                masks = masks.squeeze(1)
                metrics_res = calculate_mean_iou_boundary_accuracy(pred_masks, masks, cuda_device=f"cuda:{rank}")
                iou += metrics_res['J']["M"]
                boundary += metrics_res['F']["M"]
                overall += metrics_res['J & F']
                torch.cuda.empty_cache()
            iou /= (len(val_dataset) // world_size)
            boundary /= (len(val_dataset) // world_size)
            overall /= (len(val_dataset) // world_size)
            print(
                f"\nTest Epoch {epoch}, Rank {rank}, Mean IoU: {iou}, Mean Boundary: {boundary}, Mean Overall: {overall}")

        if epoch % Config.epoch_save_freq == 0 and rank == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(Config.save_path, f"PanoSAM2_{epoch}_{Config.backbone}.pth"))
    dist.destroy_process_group()


if __name__ == "__main__":
    train()