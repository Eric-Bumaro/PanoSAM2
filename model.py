import sys
import random
from typing import Optional, Tuple, List
import numpy as np
import math
import torch.nn as nn
from sam2.build_sam import build_sam2
import torch
import torch.nn.functional as F
from sam2.modeling.position_encoding import PositionEmbeddingRandom
from sam2.modeling.sam.transformer import Attention
from sam2.modeling.sam2_utils import LayerNorm2d, MLP, select_closest_cond_frames, get_1d_sine_pe
import CRF

random.seed(42)

def check_model(model):
    print("Model layers and their parameters:")
    for layer_name, layer in model.named_modules():
        print(f"Layer: {layer_name} ({type(layer).__name__})")
        for param_name, param in layer.named_parameters(recurse=False):  # 仅查看当前层的参数
            print(f"  Param: {param_name}")
            print(f"    Shape: {param.shape}")
            print(f"    Requires Gradient: {param.requires_grad}")
        print("-" * 40)

def init_crf(device: torch.device) -> tuple:
    params = CRF.FrankWolfeParams(
        scheme='fixed', stepsize=1.0, regularizer='l2',
        lambda_=1.0, lambda_learnable=False,
        x0_weight=0.0, x0_weight_learnable=False,
    )
    crf = CRF.DenseGaussianCRF(
        classes=2,
        alpha=24.0, beta=0.03, gamma=8.0,
        spatial_weight=3.0, bilateral_weight=20.0,
        compatibility=1.0, init='potts', solver='fw',
        iterations=10, params=params,
    ).to(device)
    return crf, 10, 0.95

@torch.no_grad()
def crf_refine(crf_module, band_px: int, p_core: float, tgt_image: torch.Tensor,
               init_mask: torch.Tensor) -> torch.Tensor:
    """Refine a binary mask using GPU CRF on a narrow band around the boundary."""
    device = tgt_image.device
    H, W = init_mask.shape

    r = max(1, band_px // 2)
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device), indexing='ij',
    )
    kernel = ((xs ** 2 + ys ** 2) <= r * r).float()[None, None]
    m0 = init_mask.float()[None, None]
    y = F.conv2d(m0, kernel, padding=r)
    ksum = float(kernel.sum().item())
    dil = (y > 0).squeeze()
    ero = (y == ksum).squeeze()
    band = dil ^ ero
    inside_core = ero
    outside_core = ~dil

    prob_fg = init_mask.float().clone()
    prob_fg[inside_core] = p_core
    prob_fg[outside_core] = 1.0 - p_core
    prob_fg[band] = 0.5

    logit_fg = torch.logit(prob_fg.clamp(1e-6, 1 - 1e-6))
    logits = torch.stack([-logit_fg, logit_fg], dim=0).unsqueeze(0).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)

    img01 = ((tgt_image * std) + mean).clamp(0, 1).unsqueeze(0)
    logits_ref = crf_module(img01.to(device=device, dtype=torch.float32), logits)

    mask_crf = logits_ref[0, 1] > logits_ref[0, 0]
    result = init_mask.clone()
    result[band] = mask_crf[band]
    result[inside_core] = True
    result[outside_core] = False
    return result


class ERPPositionEmbedding(PositionEmbeddingRandom):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__(num_pos_feats, scale)

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords @ self.positional_encoding_gaussian_matrix
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a 360° spherical image grid."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        theta = torch.linspace(-np.pi / 2, np.pi / 2, h, device=device)
        phi = torch.linspace(-np.pi, np.pi, w, device=device)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
        coords = torch.stack([theta_grid, phi_grid], dim=-1)
        pe = self._pe_encoding(coords)
        return pe.permute(2, 0, 1)

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class FiLM(nn.Module):
    def __init__(self, feature_dim, output_dim, mlp_layers: int = 4):
        super(FiLM, self).__init__()
        self.linear = MLP(feature_dim, feature_dim, output_dim * 2, num_layers=mlp_layers,
                          activation=nn.GELU, sigmoid_output=True)

    def forward(self, x):
        gamma, beta = self.linear(x).chunk(2, dim=-1)
        return gamma, beta

class ObjectAttention(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, long_obj, short_mem):
        query = self.query_proj(long_obj)
        keys = self.key_proj(short_mem)
        attn_scores = torch.matmul(query, keys.t()) / (self.feature_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return attn_weights

class LSM(nn.Module):
    def __init__(self, feature_dim: int, obj_ptr_dim: int):
        super().__init__()
        self.film = FiLM(obj_ptr_dim, feature_dim)
        self.obj_ptr_attention = ObjectAttention(obj_ptr_dim)
        initialize_weights(self)

    def forward(self, short_features, short_obj_ptrs, long_obj_ptrs):
        # [1, 7, 64, 64, 64], [1, 7, 256], [1, 2, 256]
        B, S, C_mem, H, W = short_features.shape
        _, L, _ = long_obj_ptrs.shape
        short_features = short_features.flatten(0, 1)
        short_obj_ptrs = short_obj_ptrs.flatten(0, 1)
        long_obj_ptrs = long_obj_ptrs.flatten(0, 1)
        attn_weights = self.obj_ptr_attention(long_obj_ptrs, short_obj_ptrs)
        long_features = torch.einsum("LS,SCHW->LCHW", attn_weights, short_features)
        gamma, beta = self.film(long_obj_ptrs)
        gamma = gamma.view(B * L, C_mem, 1, 1)
        beta = beta.view(B * L, C_mem, 1, 1)
        long_features = long_features * gamma + beta
        long_features = long_features.view(B, L, C_mem, H, W)
        return long_features

class PC_Block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1),
            nn.GELU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1),
            nn.GELU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=1),
            nn.GELU()
        )
        self.norm = LayerNorm2d(embed_dim)
        initialize_weights(self)


    def forward(self, x):
        conv_1 = self.conv_1(x)
        first, last = x[:, :, :, [0]].detach(), x[:, :, :, [-1]].detach()
        conv_3_input = torch.cat((last, x, first), dim=-1)
        conv_3_input = F.pad(conv_3_input, (0, 0, 1, 1), mode='constant', value=0)
        conv_3 = self.conv_3(conv_3_input)
        first_2, last_2 = x[:, :, :, :2].detach(), x[:, :, :, -2:].detach()
        conv_5_input = torch.cat((last_2, x, first_2), dim=-1)
        conv_5_input = F.pad(conv_5_input, (0, 0, 2, 2), mode='constant', value=0)
        conv_5 = self.conv_5(conv_5_input)
        out = self.norm(conv_1 + conv_3 + conv_5)
        return out + x



class PanoSAM2(nn.Module):
    def __init__(self, config):
        super(PanoSAM2, self).__init__()
        self.sam2 = build_sam2(config.sam2_config_file, ckpt_path=config.sam2_weight_file, device="cuda:0",
                               mode="train")
        self.config = config
        frozen_modules = [self.sam2.image_encoder, self.sam2.sam_prompt_encoder, self.sam2.no_mem_pos_enc,
                          self.sam2.no_mem_embed, self.sam2.mask_downsample, self.sam2.memory_encoder,
                          self.sam2.obj_ptr_proj, self.sam2.maskmem_tpos_enc, self.sam2.obj_ptr_tpos_proj,
                          self.sam2.no_obj_embed_spatial, self.sam2.no_obj_ptr, self.sam2.memory_attention,
                          self.sam2.sam_mask_decoder.conv_s0, self.sam2.sam_mask_decoder.conv_s1]
        for frozen_module in frozen_modules:
            if type(frozen_module) is nn.Parameter:
                frozen_module.requires_grad = False
            else:
                for param in frozen_module.parameters():
                    param.requires_grad = False
        # check_model(self.sam2)
        self.out_scale, self.out_bias = 20.0, -10.0
        self.output_dict = None
        self.temp_point_coords = torch.zeros(1, 1, 2)
        self.temp_point_labels = -torch.ones(1, 1, dtype=torch.int32)
        self.last_vision_feats = {}
        self.last_feat_sizes = {}
        self.last_high_res_multimask_logits = {}
        self.last_object_score = {}
        self.dense_pe = self.sam2.sam_prompt_encoder.get_dense_pe().detach()
        self.lsm = LSM(64, 256)
        self.shallow_0_pc = PC_Block(32)
        self.shallow_1_pc = PC_Block(64)
        self.deep_pc = PC_Block(256)
        self.mask_pc = PC_Block(256)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
        )

    def frames_feature_extract(self, frames):
        backbone_out = self.sam2.forward_image(frames)
        # {"vision_features": [B, 256, 64, 64], "vision_pos_enc": [[B, 256, 256, 256], [B, 256, 128, 128], [B, 256, 64, 64]],
        # "backbone_fpn": [[B, 32, 256, 256], [B, 64, 128, 128], [B, 256, 64, 64]]}
        # vision_features 和 backbone_fpn 的最后一个一样的
        _, vision_feats, vision_pos_embeds, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)
        hiera_features = [vision_feats[0].permute(1, 2, 0).view(-1, 32, *feat_sizes[0]),
                          vision_feats[1].permute(1, 2, 0).view(-1, 64, *feat_sizes[1])]
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(-1, self.sam2.hidden_dim,
                                                          *feat_sizes[-1])
        return vision_feats, vision_pos_embeds, hiera_features, pix_feat, feat_sizes

    def detach_memory(self, batch_index=0):
        self.last_vision_feats[batch_index] = [feat.detach() for feat in self.last_vision_feats[batch_index]]
        self.last_high_res_multimask_logits[batch_index] = self.last_high_res_multimask_logits[batch_index].detach()
        self.last_object_score[batch_index] = self.last_object_score[batch_index].detach()
        self.current_out[batch_index]["obj_ptr"] = self.current_out[batch_index]["obj_ptr"].detach()
        self.current_out[batch_index]["pred_masks"] = self.current_out[batch_index]["pred_masks"].detach()
        self.current_out[batch_index]["pred_masks_high_res"] = self.current_out[batch_index]["pred_masks_high_res"].detach()
        self.current_out[batch_index]["obj_score"] = self.current_out[batch_index]["obj_score"].detach()

    def occlusion_sample(self, non_cond_outputs_dict, last_frame_idx):
        long_cond_outputs= []
        obj_scores = []
        for frame_id, output in non_cond_outputs_dict.items():
            if frame_id <= last_frame_idx:
                long_cond_outputs.append((frame_id, output['obj_ptr']))
                obj_scores.append(torch.exp(output['obj_score']).item())
        if len(long_cond_outputs) == 0 or self.config.long_mem_size == 0:
            return None, None
        else:
            selected = []
            probs = np.array(obj_scores)
            probs /= probs.sum()
            if len(long_cond_outputs) < self.config.long_mem_size:
                selected = long_cond_outputs
            else:
                sampled_ids = np.random.choice(range(len(long_cond_outputs)), size=self.config.long_mem_size, p=probs, replace=False)
                for index in sampled_ids:
                    selected.append(long_cond_outputs[index])
        frame_idx, obj_ptrs = zip(*selected)
        frame_idx = list(frame_idx)
        obj_ptrs = list(obj_ptrs)
        return frame_idx, obj_ptrs

    def _prepare_memory_conditioned_features(self, frame_idx, current_vision_feats, current_vision_pos_embeds,
                                             feat_sizes, output_dict, use_long_mem=False):
        B = current_vision_feats[-1].size(1)
        C = self.sam2.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        short_oldest_allowed_idx = frame_idx - self.sam2.num_maskmem * self.config.mem_stride
        long_oldest_allowed_idx = frame_idx - self.config.long_mem_scale * self.config.mem_stride
        all_frame_idxs = list(output_dict["non_cond_frame_outputs"].keys())
        for idx in all_frame_idxs:
            if idx <= long_oldest_allowed_idx or idx % self.config.mem_stride !=0:
                output_dict["non_cond_frame_outputs"].pop(idx)
            # elif idx <= short_oldest_allowed_idx:
            #     for key in ["maskmem_features", "maskmem_pos_enc"]:
            #         if key in output_dict["non_cond_frame_outputs"][idx].keys():
            #             del output_dict["non_cond_frame_outputs"][idx][key]

        to_cat_memory, to_cat_memory_pos_embed = [], []
        t_pos_and_prevs = [(0, output_dict["cond_frame_outputs"][0])]
        mem_frame_idx = frame_idx - frame_idx % self.config.mem_stride - self.config.mem_stride
        # short_mem_features = [output_dict["cond_frame_outputs"][0]["maskmem_features"]]
        # short_obj_ptrs = [output_dict["cond_frame_outputs"][0]["obj_ptr"]]
        short_mem_features = []
        short_obj_ptrs = []
        maskmem_pos_enc = None
        for _ in range(1, self.sam2.num_maskmem):
            out = output_dict["non_cond_frame_outputs"].get(mem_frame_idx, None)
            if out is None:
                continue
            short_mem_features.append(out["maskmem_features"])
            short_obj_ptrs.append(out["obj_ptr"])
            if maskmem_pos_enc is None:
                maskmem_pos_enc = [temp.clone() for temp in out["maskmem_pos_enc"]]
            t_pos_and_prevs.append((mem_frame_idx, out))
            mem_frame_idx -= self.config.mem_stride
        long_mem_idx, long_mem_obj_pts = self.occlusion_sample(output_dict["non_cond_frame_outputs"], mem_frame_idx)
        long_mem_features, gt_long_mem_features = None, None
        if long_mem_idx is not None and use_long_mem:
            gt_long_mem_features = []
            for long_mem_idx_ in long_mem_idx:
                gt_long_mem_features.append(output_dict["non_cond_frame_outputs"][long_mem_idx_]["maskmem_features"])
            gt_long_mem_features = torch.stack(gt_long_mem_features, dim=1)
            short_mem_features = torch.stack(short_mem_features, dim=1)
            short_obj_ptrs = torch.stack(short_obj_ptrs, dim=1)
            long_mem_obj_pts = torch.stack(long_mem_obj_pts, dim=1)
            long_mem_features = self.lsm(short_mem_features, short_obj_ptrs, long_mem_obj_pts)
            for long_index in range(long_mem_obj_pts.shape[1]):
                out = {
                    "maskmem_features": long_mem_features[:, long_index],
                    "maskmem_pos_enc": maskmem_pos_enc,
                    "obj_ptr": long_mem_obj_pts[:, long_index]
                }
                t_pos_and_prevs.append((long_mem_idx[long_index], out))
        short_size = min(len(t_pos_and_prevs), self.sam2.num_maskmem)
        for index, (t, prev) in enumerate(t_pos_and_prevs):
            feats = prev["maskmem_features"].to(device, non_blocking=True)
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
            maskmem_enc = prev["maskmem_pos_enc"][-1]
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
            if t == 0:
                maskmem_enc = (
                        maskmem_enc + self.sam2.maskmem_tpos_enc[6]
                )
            elif t > short_oldest_allowed_idx:
                maskmem_enc = (
                        maskmem_enc + self.sam2.maskmem_tpos_enc[short_size - index - 1]
                )
            else:
                maskmem_enc = (
                        maskmem_enc + self.sam2.maskmem_tpos_enc[self.sam2.num_maskmem - index - 2]
                )
            to_cat_memory_pos_embed.append(maskmem_enc)

        pos_and_ptrs = []
        for t_pos, prev  in t_pos_and_prevs:
            t = frame_idx - t_pos
            pos_and_ptrs.insert(1, (t, prev["obj_ptr"]))
        pos_list, ptrs_list = zip(*pos_and_ptrs)
        obj_ptrs = torch.stack(ptrs_list, dim=0)
        t_diff_max = self.sam2.num_maskmem + self.config.long_mem_size
        obj_pos = torch.tensor(pos_list).to(device=device, non_blocking=True)
        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=self.sam2.hidden_dim)
        obj_pos = self.sam2.obj_ptr_tpos_proj(obj_pos)
        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.sam2.mem_dim)
        if self.sam2.mem_dim < C:
            obj_ptrs = obj_ptrs.reshape(-1, B, C // self.sam2.mem_dim, self.sam2.mem_dim)
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = obj_pos.repeat_interleave(C // self.sam2.mem_dim, dim=0)
        to_cat_memory.append(obj_ptrs)
        to_cat_memory_pos_embed.append(obj_pos)
        num_obj_ptr_tokens = obj_ptrs.shape[0]
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        pix_feat_with_mem = self.sam2.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            high_res_features: List[torch.Tensor]
        ):
        output_tokens = torch.cat(
            [
                self.sam2.sam_mask_decoder.obj_score_token.weight,
                self.sam2.sam_mask_decoder.iou_token.weight,
                self.sam2.sam_mask_decoder.mask_tokens.weight,
            ],
            dim=0,
        )
        s = 1
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        assert image_embeddings.shape[0] == tokens.shape[0]
        src = image_embeddings
        src = src + dense_prompt_embeddings

        assert (image_pe.size(0) == 1), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.sam2.sam_mask_decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1: (s + 1 + self.sam2.sam_mask_decoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        dc1, ln1, act1, dc2, act2 = self.sam2.sam_mask_decoder.output_upscaling
        feat_s0, feat_s1 = high_res_features
        feat_s0 = self.shallow_0_pc(feat_s0)
        feat_s1 = self.shallow_1_pc(feat_s1)
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.sam2.sam_mask_decoder.num_mask_tokens):
            hyper_in_list.append(
                self.sam2.sam_mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.sam2.sam_mask_decoder.iou_prediction_head(iou_token_out)
        object_score_logits = self.sam2.sam_mask_decoder.pred_obj_score_head(hs[:, 0, :])
        masks = masks[:, 1:, :, :]
        iou_pred = iou_pred[:, 1:]
        sam_tokens_out = mask_tokens_out[:, 1:]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    @torch.no_grad()
    def process_first_frame(self, first_frame, first_mask, batch_index=0):
        if batch_index == 0:
            self.output_dict = {}
            self.current_out = {}
        self.output_dict[batch_index] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self.temp_point_coords = self.temp_point_coords.to(first_frame.device)
        self.temp_point_labels = self.temp_point_labels.to(first_frame.device)
        self.dense_pe = self.dense_pe.to(first_frame.device)
        self.current_out[batch_index] = {"point_inputs": None, "mask_inputs": None, "prototype": None}
        vision_feats, _, hiera_features, pix_feat, feat_sizes = self.frames_feature_extract(first_frame)
        low_mask_prompt = self.sam2.mask_downsample(first_mask)
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=(self.temp_point_coords, self.temp_point_labels),
            boxes=None,
            masks=low_mask_prompt,
        )
        low_res_multimasks_logits, ious, sam_output_tokens, _ = self.sam2.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=self.dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=hiera_features,
            multimask_output=True,
            repeat_image=False
        )
        max_iou_index = ious.argmax(dim=-1)
        sam_output_token = sam_output_tokens[[0], max_iou_index]
        high_res_masks = first_mask * self.out_scale + self.out_bias
        low_res_multimask_logits = F.interpolate(high_res_masks, size=(256, 256), align_corners=False,
                                                 mode="bilinear", antialias=True)
        is_obj_appearing = torch.any(low_res_multimask_logits.flatten(1).float() > 0.0, dim=1)[..., None]
        object_score_logits = self.out_scale * is_obj_appearing.float() + self.out_bias
        obj_ptr = self.sam2.obj_ptr_proj(sam_output_token)
        last_object_score = object_score_logits.sigmoid()
        obj_ptr = obj_ptr + (1 - last_object_score) * self.sam2.no_obj_ptr
        self.current_out[batch_index]["pred_masks"] = low_res_multimask_logits
        self.current_out[batch_index]["pred_masks_high_res"] = high_res_masks
        self.current_out[batch_index]["obj_ptr"] = obj_ptr
        self.current_out[batch_index]["obj_score"] = last_object_score
        self.last_feat_sizes[batch_index] = feat_sizes
        self.last_vision_feats[batch_index] = vision_feats
        self.last_high_res_multimask_logits[batch_index] = high_res_masks
        self.last_object_score[batch_index] = last_object_score
        return low_res_multimask_logits

    def process_frame(self, frame, frame_id, use_long_mem=False, gt_mask=None, correct_freq=None, batch_index=0):
        frame_vision_feats, frame_pos_embeds, frame_hiera_features, _, feat_sizes = self.frames_feature_extract(frame)
        self.detach_memory(batch_index)
        maskmem_features, maskmem_pos_enc = self.sam2._encode_new_memory(
            current_vision_feats=self.last_vision_feats[batch_index],
            feat_sizes=self.last_feat_sizes[batch_index],
            pred_masks_high_res=self.last_high_res_multimask_logits[batch_index],
            object_score_logits=self.last_object_score[batch_index],
            is_mask_from_pts=None,
        )
        self.current_out[batch_index]["maskmem_features"] = maskmem_features.detach()
        for i in range(len(maskmem_pos_enc)):
            maskmem_pos_enc[i] = maskmem_pos_enc[i].detach()
        self.current_out[batch_index]["maskmem_pos_enc"] = maskmem_pos_enc
        if frame_id - 1 == 0:
            self.output_dict[batch_index]["cond_frame_outputs"][frame_id - 1] = self.current_out[batch_index]
        else:
            self.output_dict[batch_index]["non_cond_frame_outputs"][frame_id - 1] = self.current_out[batch_index]
        self.current_out[batch_index] = {"point_inputs": None, "mask_inputs": None, "prototype": None}
        # [1, 32, 256, 256],[1, 64, 128, 128]
        pix_feat = self._prepare_memory_conditioned_features(
            frame_idx=frame_id,
            current_vision_feats=frame_vision_feats[-1:],
            current_vision_pos_embeds=frame_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=self.output_dict[batch_index],
            use_long_mem=use_long_mem
        ) #[1, 256, 64, 64]
        pix_feat = self.deep_pc(pix_feat)
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=(self.temp_point_coords, self.temp_point_labels),
            boxes=None,
            masks=None,
        )
        last_mask_embedding = self.sam2.memory_encoder.mask_downsampler(self.last_high_res_multimask_logits[batch_index])
        last_mask_embedding = self.mask_pc(last_mask_embedding)
        pix_feat = self.conv(torch.cat([pix_feat, last_mask_embedding], dim=1))
        sam_low_res_multimasks_logits, ious, sam_output_tokens, object_score_logits = self.predict_masks(
            image_embeddings=pix_feat,
            image_pe=self.dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=frame_hiera_features
        ) # [1, 1, 256, 256]
        max_iou_index = ious.argmax(dim=-1)
        sam_low_res_mask_logit = sam_low_res_multimasks_logits[:, max_iou_index]
        sam_output_token = sam_output_tokens[[0], max_iou_index]
        if gt_mask is not None and (frame_id - 1) % correct_freq == 0:
            gt_mask = gt_mask * self.out_scale + self.out_bias
            sam_low_res_mask_logit_mem = F.interpolate(gt_mask, size=(256, 256), align_corners=False,
                                                     mode="bilinear", antialias=True)
            sam_high_res_mask_logit_mem = gt_mask
        else:
            # is_obj_appearing = object_score_logits > 0
            # sam_low_res_mask_logit_mem = torch.where(
            #     is_obj_appearing[:, None, None],
            #     sam_low_res_mask_logit,
            #     -1024,
            # )
            sam_low_res_mask_logit_mem = sam_low_res_mask_logit
            sam_high_res_mask_logit_mem = F.interpolate(sam_low_res_mask_logit_mem, size=(1024, 1024), mode="bilinear",
                                                         align_corners=False)
        obj_ptr = self.sam2.obj_ptr_proj(sam_output_token)
        object_score = object_score_logits.sigmoid()
        obj_ptr = obj_ptr + (1 - object_score) * self.sam2.no_obj_ptr
        self.current_out[batch_index]["pred_masks"] = sam_low_res_mask_logit_mem
        self.current_out[batch_index]["pred_masks_high_res"] = sam_high_res_mask_logit_mem
        self.current_out[batch_index]["obj_ptr"] = obj_ptr
        self.current_out[batch_index]["obj_score"] = object_score
        self.last_feat_sizes[batch_index] = feat_sizes
        self.last_vision_feats[batch_index] = frame_vision_feats
        self.last_object_score[batch_index] = object_score
        self.last_high_res_multimask_logits[batch_index] = sam_high_res_mask_logit_mem
        if not self.training:
            return sam_low_res_mask_logit
        else:
            return sam_low_res_multimasks_logits, ious, object_score_logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = torch.rand((1, 7, 64, 64, 64)).to(device)
    short_obj_pts = torch.rand((1, 7, 256)).to(device)
    long_obj_pts = torch.rand((1, 2, 256)).to(device)
    lsm = LSM(64, 256).to(device)
    lsm.eval()
    with torch.no_grad():
        output = lsm(feature, short_obj_pts, long_obj_pts)
        print(output.shape)
