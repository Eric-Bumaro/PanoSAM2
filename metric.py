import math
import warnings
import torch
import cv2

def db_eval_iou(annotation, segmentation, void_pixels):
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.bool()
    segmentation = segmentation.bool()
    inters = torch.sum((segmentation & annotation) & ~void_pixels, dim=(-2, -1))
    union = torch.sum((segmentation | annotation) & ~void_pixels, dim=(-2, -1))
    j = inters / union
    if j.ndim == 0:
        j = 1 if torch.isclose(union, torch.tensor(0, dtype=union.dtype)) else j
    else:
        j[torch.isclose(union, torch.tensor(0, dtype=union.dtype))] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels, cuda_device, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = torch.zeros(n_frames)
        f_res = f_measure(segmentation, annotation, void_pixels, cuda_device, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, cuda_device, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels, cuda_device, bound_th=0.008):
    """
    Compute mean, recall, and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (torch.Tensor): binary segmentation image.
        gt_mask         (torch.Tensor): binary annotated image.
        void_pixels     (torch.Tensor): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    void_pixels = void_pixels.bool()
    bound_pix = bound_th if bound_th >= 1 else \
        math.ceil(bound_th * torch.norm(torch.tensor(foreground_mask.shape[:2], dtype=torch.float)))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * ~void_pixels)
    gt_boundary = _seg2bmap(gt_mask * ~void_pixels)

    # Perform dilation using PyTorch
    fg_dil = torch_dilate(fg_boundary, bound_pix, cuda_device)
    gt_dil = torch_dilate(gt_boundary, bound_pix, cuda_device)

    # Get the intersection
    gt_match = gt_boundary & fg_dil
    fg_match = fg_boundary & gt_dil

    # Area of the intersection
    n_fg = torch.sum(fg_boundary, dim=(-2, -1))
    n_gt = torch.sum(gt_boundary, dim=(-2, -1))

    precision = torch.sum(fg_match, dim=(-2, -1))/ n_fg
    recall = torch.sum(gt_match, dim=(-2, -1)) / n_gt

    precision[(n_fg == 0) & (n_gt > 0)] = 1
    recall[(n_fg == 0) & (n_gt > 0)] = 0

    precision[(n_fg > 0) & (n_gt == 0)] = 0
    recall[(n_fg > 0) & (n_gt == 0)] = 1

    precision[(n_fg == 0) & (n_gt == 0)] = 1
    recall[(n_fg == 0) & (n_gt == 0)] = 1


    # Compute F measure
    F = 2 * precision * recall / (precision + recall)
    F[precision + recall == 0] = 0

    return F


def torch_dilate(mask, bound_pix, cuda_device, batch_size=300):
    """
    Perform dilation using PyTorch's convolution operation.

    Arguments:
        mask (torch.Tensor): Binary mask (2D tensor).
        bound_pix (int): Radius of the dilation disk.

    Returns:
        dilated_mask (torch.Tensor): Dilated binary mask.
    """
    # Create a circular kernel (disk) for dilation
    y, x = torch.meshgrid(torch.arange(-bound_pix, bound_pix + 1), torch.arange(-bound_pix, bound_pix + 1))
    kernel = (x**2 + y**2 <= bound_pix**2).float().to(cuda_device).unsqueeze(0).unsqueeze(0)

    dilated_masks = []

    for i in range(0, mask.size(0), batch_size):
        mask_batch = mask[i:i + batch_size].unsqueeze(1).float().to(cuda_device)
        dilated_mask_batch = torch.nn.functional.conv2d(mask_batch, kernel, padding=bound_pix)
        dilated_mask_batch = (dilated_mask_batch > 0).squeeze(1)
        dilated_masks.append(dilated_mask_batch.cpu())
    dilated_masks = torch.cat(dilated_masks, dim=0)
    return dilated_masks.detach().cpu()


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries. The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width   : Width of desired bmap  <= seg.shape[1]
        height  : Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (torch.Tensor): Binary boundary map.
    """
    seg = seg.bool()
    seg[seg > 0] = 1

    width = seg.shape[2] if width is None else width
    height = seg.shape[1] if height is None else height

    h, w = seg.shape[1:3]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w or height > h or abs(ar1 - ar2) > 0.01
    ), f"Can't convert {w}x{h} seg to {width}x{height} bmap."

    e = torch.zeros_like(seg)
    s = torch.zeros_like(seg)
    se = torch.zeros_like(seg)

    e[:, :, :-1] = seg[:, :, 1:]
    s[:, :-1, :] = seg[:, 1:, :]
    se[:, :-1, :-1] = seg[:, 1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[:, -1, :] = seg[:, -1, :] ^ e[:, -1, :]
    b[:, :, -1] = seg[:, :, -1] ^ s[:, :, -1]
    b[:, -1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        j_coords, i_coords = torch.meshgrid(
            torch.arange(0, h, device=seg.device),
            torch.arange(0, w, device=seg.device),
            indexing='ij'
        )

        # Scale coordinates to the new size
        j_scaled = ((j_coords.float() + 1) * height / h).floor().long()
        i_scaled = ((i_coords.float() + 1) * width / w).floor().long()

        # Initialize the boundary map
        bmap = torch.zeros((len(seg), height, width), device=seg.device, dtype=seg.dtype)

        # Apply the boundary map using broadcasting
        bmap[:, j_scaled[b], i_scaled[b]] = 1

    return bmap


def _evaluate_semisupervised(pred_masks, gt_masks, cuda_device):
    void_pixels = torch.zeros_like(pred_masks, dtype=torch.bool).to(pred_masks.device)
    j_metrics_res = db_eval_iou(gt_masks, pred_masks, void_pixels)
    f_metrics_res = db_eval_boundary(gt_masks, pred_masks, void_pixels, cuda_device)
    return j_metrics_res, f_metrics_res


def db_statistics(per_frame_values):
    """Compute mean, recall, and decay from per-frame evaluation.

    Arguments:
        per_frame_values (torch.Tensor): Per-frame evaluation.

    Returns:
        M, O, D (float, float, float):
            Return evaluation statistics: mean, recall, decay.
    """
    # Strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = torch.nanmean(per_frame_values)
        O = torch.nanmean((per_frame_values > 0.5).float())

    N_bins = 4
    ids = torch.round(torch.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.long()

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = torch.nanmean(D_bins[0]) - torch.nanmean(D_bins[3])

    return M.item(), O.item(), D.item()


def calculate_mean_iou_boundary_accuracy(pred_masks, gt_masks, cuda_device):
    assert pred_masks.shape == gt_masks.shape, "pred_masks and gt_masks must have the same shape"
    metrics_res = {"J": {}, "F": {}}
    j_metrics_res, f_metrics_res = _evaluate_semisupervised(gt_masks, pred_masks, cuda_device)
    [JM, JR, JD] = db_statistics(j_metrics_res)
    metrics_res['J']["M"] = JM
    metrics_res['J']["R"] = JR
    metrics_res['J']["D"] = JD
    [FM, FR, FD] = db_statistics(f_metrics_res)
    metrics_res['F']["M"] = FM
    metrics_res['F']["R"] = FR
    metrics_res['F']["D"] = FD
    metrics_res['J & F'] = (JM + FM) / 2
    return metrics_res

def train_val_iou_measure(pred_masks, gt_masks):
    pred_masks = pred_masks.float().flatten(0, 2)
    gt_masks = gt_masks.float().flatten(0, 2)
    intersection = (pred_masks * gt_masks).sum(dim=(-1, -2))
    pred_masks_sum = pred_masks.sum(dim=(-1, -2))
    gt_masks_sum = gt_masks.sum(dim=(-1, -2))
    union = pred_masks_sum + gt_masks_sum - intersection
    union[union == 0] = 1
    intersection[union == 0] = 1
    mean_iou = (intersection / union).mean().item()
    return mean_iou

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_masks = torch.randint(0, 2, (10, 256, 256), dtype=torch.uint8).to(device)
    gt_masks = pred_masks.clone()
    metrics_res = calculate_mean_iou_boundary_accuracy(pred_masks, gt_masks, device)
    print(metrics_res)
