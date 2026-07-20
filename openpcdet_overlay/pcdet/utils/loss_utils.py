import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def neg_loss_ras(pred, gt, spatial_mask=None, quality_scores=None, pos_inds=None, obj_mask=None, beta=2.0, alpha=2.0, gamma=4.0):
    """
    RAS (Rank-aware Adaptive Self-Distillation) Focal Loss for SR3D
    """
    pos_mask = gt.eq(1)
    neg_mask = gt.lt(1)

    neg_weights = torch.pow(1 - gt, gamma)

    # =========================================================
    # 核心映射逻辑：将 1D 的质量得分转换到 2D 空间特征图上
    # =========================================================
    target_soft = gt.clone()
    if quality_scores is not None and pos_inds is not None and obj_mask is not None:
        B, C, H, W = gt.shape
        
        # 1. 计算软标签 y_soft = IoU^beta
        soft_labels = torch.pow(quality_scores, beta) # (B, MAX_OBJS)
        
        # 2. 创建一个空间软标签图 (B, H*W)，默认全 1
        spatial_soft_map = torch.ones((B, H * W), dtype=gt.dtype, device=gt.device)
        
        # 3. 将计算好的软标签根据正样本的索引 scatter 进去
        for b in range(B):
            valid = obj_mask[b].bool() # 过滤掉 padding 的空目标
            if valid.sum() > 0:
                inds_b = pos_inds[b, valid].long()
                vals_b = soft_labels[b, valid]
                spatial_soft_map[b, inds_b] = vals_b
                
        # 4. 扩展维度并替换掉 gt==1 处的值
        spatial_soft_map = spatial_soft_map.view(B, 1, H, W).expand(B, C, H, W)
        target_soft[pos_mask] = spatial_soft_map[pos_mask]
    # =========================================================

    # 计算损失：正样本的靶点变成了 target_soft，惩罚预测值与真实定位质量的差距
    pos_loss = torch.log(pred) * torch.pow(torch.abs(target_soft - pred), alpha) * pos_mask.float()
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask.float()

    if spatial_mask is not None:
        spatial_mask = spatial_mask[:, None, :, :].float()
        pos_loss = pos_loss * spatial_mask
        neg_loss = neg_loss * spatial_mask
        num_pos = (pos_mask.float() * spatial_mask).sum()
    else:
        num_pos = pos_mask.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
        
    return loss

def neg_loss_sparse(pred, gt):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x n)
        gt: (batch x c x n)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Modified Focal Loss to support RAS (Rank-aware Adaptive Self-Distillation)
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet
        self.ras_loss = neg_loss_ras # 引入我们刚写的 RAS Loss

    def forward(self, out, target, mask=None, quality_scores=None, pos_inds=None):
        spatial_mask = None
        obj_mask = None
        
        # 安全处理：区分传进来的是 2D 的对象掩码还是 3D 的空间掩码
        if mask is not None:
            if mask.dim() == 2: # (B, MAX_OBJS)
                obj_mask = mask
            else:
                spatial_mask = mask # (B, H, W)

        # 触发 RAS 训练机制
        if quality_scores is not None and pos_inds is not None and obj_mask is not None:
            return self.ras_loss(
                out, target, 
                spatial_mask=spatial_mask, 
                quality_scores=quality_scores, 
                pos_inds=pos_inds, 
                obj_mask=obj_mask
            )
        else:
            # 退化为标准的 CenterNet Focal Loss (推理时或未开启 RAS 时)
            return self.neg_loss(out, target, mask=spatial_mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class FocalLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossSparse, self).__init__()
        self.neg_loss = neg_loss_sparse

    def forward(self, out, target):
        return self.neg_loss(out, target)


def focal_loss_sparse(pred, target, eps=1e-4):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, min=eps, max=1 - eps)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


class DynamicPositiveMask(nn.Module):
    def __init__(self, cls_weight=1, reg_weight=2, voxel_size=(0.8, 0.8)):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.voxel_size = voxel_size

    @staticmethod
    def cls_cost(pred_cls, pos_mask):
        return 1 - torch.max(pred_cls * pos_mask, dim=-1)[0]

    def rwiou_cost(self, pred_reg, gt_reg, mask, r_factor=0.2):
        is_finite = torch.isfinite(gt_reg).all(dim=-1)
        u, rdiou = box_utils.get_rwiou(pred_reg, gt_reg, r_factor, self.voxel_size)
        reg_cost = 1 - torch.clamp(rdiou, min=0, max=1) + u
        return reg_cost * mask * is_finite, u

    def forward(self, pred_cls, target_cls, pred_reg, gt_reg, masks, iou_target, r_factor=0.5):
        with torch.no_grad():
            cls_cost = self.cls_cost(pred_cls, target_cls)
            reg_cost, _ = self.rwiou_cost(pred_reg, gt_reg, masks, r_factor)

            all_cost = (
                self.cls_weight * cls_cost * masks
                + self.reg_weight * reg_cost
                + (1 - masks.float()) * 100
            )
            local_sort_inds = torch.sort(all_cost, dim=-1)[1]
            positive_nums = torch.sum(iou_target, dim=-1).clamp(min=1).int()
            positive_masks = target_cls.new_zeros(*masks.shape)

            for batch_id in range(pred_cls.shape[0]):
                box_num = (torch.sum(masks[batch_id], dim=-1) > 0).sum()
                for box_id in range(box_num):
                    count = positive_nums[batch_id, box_id]
                    local_pos_inds = local_sort_inds[batch_id, box_id, :count]
                    positive_masks[batch_id, box_id] = iou_target[batch_id, box_id]
                    positive_masks[batch_id, box_id, local_pos_inds] = 1

        return positive_masks * masks


class SlotFormerIoULoss(nn.Module):
    def forward(self, pred_ious, gt_ious, mask):
        mask = mask.bool()
        loss = F.l1_loss(pred_ious[mask], gt_ious[mask], reduction='sum')
        return loss / torch.clamp_min(mask.sum(), min=1.0)


class RWIoULoss(nn.Module):
    def __init__(self, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size

    def forward(self, pred_reg, gt_reg, mask, r_factor=0.5):
        is_finite = torch.isfinite(gt_reg).all(dim=-1)
        u, rwiou = box_utils.get_rwiou(pred_reg, gt_reg, r_factor, self.voxel_size)
        loss = (1 - torch.clamp(rwiou, min=0, max=1) + u) * mask * is_finite
        return loss.sum() / torch.clamp_min((mask * is_finite).sum(), min=1.0)


class RegLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossSparse, self).__init__()

    def forward(self, output, mask, ind=None, target=None, batch_index=None):
        """
        Args:
            output: (N x dim)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """

        pred = []
        batch_size = mask.shape[0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred.append(output[batch_inds][ind[bs_idx]])
        pred = torch.stack(pred)

        loss = _reg_loss(pred, target, mask)
        return loss


class IouLossSparse(nn.Module):
    '''IouLoss loss for an output tensor
    Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(IouLossSparse, self).__init__()

    def forward(self, iou_pred, mask, ind, box_pred, box_gt, batch_index):
        if mask.sum() == 0:
            return iou_pred.new_zeros((1))
        batch_size = mask.shape[0]
        mask = mask.bool()

        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred = iou_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            pred_box = box_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            target = iou3d_nms_utils.boxes_aligned_iou3d_gpu(pred_box, box_gt[bs_idx])
            target = 2 * target - 1
            loss += F.l1_loss(pred, target, reduction='sum')

        loss = loss / (mask.sum() + 1e-4)
        return loss

class IouRegLossSparse(nn.Module):
    '''Distance IoU loss for output boxes
        Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
    '''

    def __init__(self, type="DIoU"):
        super(IouRegLossSparse, self).__init__()

    def center_to_corner2d(self, center, dim):
        corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                    dtype=torch.float32, device=dim.device)
        corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
        corners = corners + center.view(-1, 1, 2)
        return corners

    def bbox3d_iou_func(self, pred_boxes, gt_boxes):
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        qcorners = self.center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
        gcorners = self.center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

        inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
        inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
        out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
        out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

        # calculate area
        volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
        volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

        inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
                torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
        inter_h = torch.clamp(inter_h, min=0)

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        volume_inter = inter[:, 0] * inter[:, 1] * inter_h
        volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

        # boxes_iou3d_gpu(pred_boxes, gt_boxes)
        inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

        outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
                torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
        outer_h = torch.clamp(outer_h, min=0)
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

        dious = volume_inter / volume_union - inter_diag / outer_diag
        dious = torch.clamp(dious, min=-1.0, max=1.0)

        return dious

    def forward(self, box_pred, mask, ind, box_gt, batch_index):
        if mask.sum() == 0:
            return box_pred.new_zeros((1))
        mask = mask.bool()
        batch_size = mask.shape[0]

        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred_box = box_pred[batch_inds][ind[bs_idx]]
            iou = self.bbox3d_iou_func(pred_box[mask[bs_idx]], box_gt[bs_idx])
            loss += (1. - iou).sum()

        loss =  loss / (mask.sum() + 1e-4)
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
       
    def forward(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0
        assert pred.size() == target.size()
        loss = torch.abs(pred - target)
        return loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights

        return pos_loss + neg_loss


def calculate_iou_loss_centerhead(iou_preds, batch_box_preds, mask, ind, gt_boxes):
    """
    Args:
        iou_preds: (batch x 1 x h x w)
        batch_box_preds: (batch x (7 or 9) x h x w)
        mask: (batch x max_objects)
        ind: (batch x max_objects)
        gt_boxes: (batch x N, 7 or 9)
    Returns:
    """
    if mask.sum() == 0:
        return iou_preds.new_zeros((1))

    mask = mask.bool()
    selected_iou_preds = _transpose_and_gather_feat(iou_preds, ind)[mask]

    selected_box_preds = _transpose_and_gather_feat(batch_box_preds, ind)[mask]
    iou_target = iou3d_nms_utils.paired_boxes_iou3d_gpu(selected_box_preds[:, 0:7], gt_boxes[mask][:, 0:7])
    # iou_target = iou3d_nms_utils.boxes_iou3d_gpu(selected_box_preds[:, 0:7].clone(), gt_boxes[mask][:, 0:7].clone()).diag()
    iou_target = iou_target * 2 - 1  # [0, 1] ==> [-1, 1]

    # print(selected_iou_preds.view(-1), iou_target)
    loss = F.l1_loss(selected_iou_preds.view(-1), iou_target, reduction='sum')
    loss = loss / torch.clamp(mask.sum(), min=1e-4)
    return loss


def calculate_iou_reg_loss_centerhead(batch_box_preds, mask, ind, gt_boxes):
    if mask.sum() == 0:
        return batch_box_preds.new_zeros((1))

    mask = mask.bool()

    selected_box_preds = _transpose_and_gather_feat(batch_box_preds, ind)

    iou = box_utils.bbox3d_overlaps_diou(selected_box_preds[mask][:, 0:7], gt_boxes[mask][:, 0:7])

    loss = (1.0 - iou).sum() / torch.clamp(mask.sum(), min=1e-4)
    return loss
