import torch
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize

##########################################################################################################################################

def focal_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):

    pt = torch.where(target == 1, pred, 1-pred)
    ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
    focal_term = (1 - pt).pow(gamma)
    loss = alpha * focal_term * ce_loss

    return loss.mean()


def dice_loss(pred, target, smooth=1.0):

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()

    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def compute_loss(pred_mask, target_mask, pred_iou, true_iou):

    pred_mask = F.sigmoid(pred_mask).squeeze(1).to(dtype = torch.float32)
    fl = focal_loss(pred_mask, target_mask)
    dl = dice_loss(pred_mask, target_mask)
    mask_loss = 20 * fl + dl
    iou_loss = F.mse_loss(pred_iou, true_iou)
    total_loss = mask_loss + iou_loss

    return total_loss


def mean_iou(preds, labels, eps=1e-6):
    
    preds = normalize(threshold(preds, 0.0, 0)).squeeze(1)
    pred_cls = (preds == 1).float()
    label_cls = (labels == 1).float()
    intersection = (pred_cls * label_cls).sum(1).sum(1)
    union = (1 - (1 - pred_cls) * (1 - label_cls)).sum(1).sum(1)
    intersection = intersection + (union == 0)
    union = union + (union == 0)
    ious = intersection / union

    return ious